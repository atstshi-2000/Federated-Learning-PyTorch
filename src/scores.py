import copy
import torch
#from .metrics import cross_entropy_loss
#import flax.linen as nn
#from jax import jacrev, jit, vmap
#import jax.numpy as jnp
import numpy as np


def cross_entropy_loss(logits, labels):
  return torch.mean(-torch.sum(torch.nn.functional.log_softmax(logits) * labels, axis=-1))
###data_diet gradients.py
def flatten_jacobian(J):
  """Jacobian pytree -> Jacobian matrix"""
  return torch.concatenate(tree_flatten(tree_map(vmap(jnp.ravel), J))[0], axis=1)


def get_mean_logit_gradients_fn(fn, params, state):
  """fn, params, state -> (X -> mean logit gradients of fn(X; params, state))"""
  return lambda X: flatten_jacobian(torch.func.jacrev(lambda p, x: fn(p, state, x).mean(0))(params, X))


def compute_mean_logit_gradients(fn, params, state, X, batch_sz):
  """compute_mean_logit_gradients: fn, params, state, X, batch_sz -> mlg
  In:
    fn      : func           : params, state, X -> logits of X at (params, state)
    params  : pytree         : parameters
    state   : pytree         : model state
    X       : nparr(n, image): images
    batch_sz: int            : image batch size for computation
  Out:
    mlgs: nparr(n_cls, n_params): mean logit gradients
  """
  # batch data for computation
  n_batches = X.shape[0] // batch_sz
  Xs = np.split(X, n_batches)
  # compute mean logit gradients
  mean_logit_gradients = jit(get_mean_logit_gradients_fn(fn, params, state))
  mlgs = 0
  for X in tqdm(Xs):
    mlgs += np.array(mean_logit_gradients(X)) / n_batches
  return mlgs


###data_diet score.py###
def get_class_balanced_random_subset(X, Y, cls_smpl_sz, seed):
  """get_class_balanced_random_subset: X, Y, cls_smpl_sz, seed -> X, Y
  In:
    X          : nparr(N, img): all images, ASSUME sorted by class
    Y          : nparr(N, C)  : corresponding labels, ASSUME equal number of examples per class
    cls_smpl_sz: int          : number of examples per class in subset
    seed       : int          : random seed
  Out:
    X: nparr(C * cls_smpl_sz, img): subsampled images, cls_smpl_sz examples per class, sorted by class
    Y: nparr(C * cls_smpl_sz, C)  : corresponding labels
  """
  # reshape to class x batch x image/label
  n_cls = Y.shape[1]
  X_c, Y_c = np.stack(np.split(X, n_cls)), np.stack(np.split(Y, n_cls))
  # sample from batch dimension
  rng = np.random.RandomState(seed)
  idxs = [rng.choice(X_c.shape[1], cls_smpl_sz, replace=False) for _ in range(n_cls)]
  X = np.concatenate([X_c[c, idxs[c]] for c in range(n_cls)])
  Y = np.concatenate([Y_c[c, idxs[c]] for c in range(n_cls)])
  return X, Y

def get_lord_error_fn(fn, params, state, ord):
  def lord_error(X, Y):
    errors = nn.softmax(fn(params, state, X)) - Y
    scores = torch.linalg.norm(errors, ord=ord, axis=-1)
    return scores
  np_lord_error = lambda X, Y: np.array(lord_error(X, Y))
  return np_lord_error


def get_margin_error(fn, params, state, score_type):
  fn_jit = jit(lambda X: fn(params, state, X))

  def margin_error(X, Y):
    batch_sz = X.shape[0]
    P = np.array(nn.softmax(fn_jit(X)))
    correct_logits = Y.astype(bool)
    margins = P[~correct_logits].reshape(batch_sz, -1) - P[correct_logits].reshape(batch_sz, 1)
    if score_type == 'max':
      scores = np.max(margins, -1)
    elif score_type == 'sum':
      scores = np.sum(margins, -1)
    return scores

  return margin_error


def get_grad_norm_fn(fn, params, state):
  def score_fn(X, Y):
    per_sample_loss_fn = lambda p, x, y: vmap(cross_entropy_loss)(fn(p, state, x), y)
    loss_grads = flatten_jacobian(jacrev(per_sample_loss_fn)(params, X, Y))
    scores = torch.linalg.norm(loss_grads, axis=-1)
    return scores

  return lambda X, Y: np.array(score_fn(X, Y))


def get_score_fn(fn, params, state, score_type):
  if score_type == 'l2_error':
    print(f'compute {score_type}...')
    score_fn = get_lord_error_fn(fn, params, state, 2)
  elif score_type == 'l1_error':
    print(f'compute {score_type}...')
    score_fn = get_lord_error_fn(fn, params, state, 1)
  elif score_type == 'max_margin':
    print(f'compute {score_type}...')
    score_fn = get_margin_error(fn, params, state, 'max')
  elif score_type == 'sum_margin':
    print(f'compute {score_type}...')
    score_fn = get_margin_error(fn, params, state, 'sum')
  elif score_type == 'grad_norm':
    print(f'compute {score_type}...')
    score_fn = get_grad_norm_fn(fn, params, state)
  else:
    raise NotImplementedError
  return score_fn


def compute_scores(fn, params, state, X, Y, batch_sz, score_type):
  n_batches = X.shape[0] // batch_sz
  Xs, Ys = np.split(X, n_batches), np.split(Y, n_batches)
  score_fn = get_score_fn(fn, params, state, score_type)
  scores = []
  for i, (X, Y) in enumerate(zip(Xs, Ys)):
    print(f'score batch {i+1} of {n_batches}')
    scores.append(score_fn(X, Y))
  scores = np.concatenate(scores)
  return scores


def compute_unclog_scores(fn, params, state, X, Y, cls_smpl_sz, seed, batch_sz_mlgs):
  n_batches = X.shape[0]
  Xs = np.split(X, n_batches)
  X_mlgs, _ = get_class_balanced_random_subset(X, Y, cls_smpl_sz, seed)
  mlgs = compute_mean_logit_gradients(fn, params, state, X_mlgs, batch_sz_mlgs)
  logit_grads_fn = get_mean_logit_gradients_fn(fn, params, state)
  score_fn = jit(lambda X: torch.linalg.norm((logit_grads_fn(X) - mlgs).sum(0)))
  scores = []
  for i, X in enumerate(Xs):
    if i % 500 == 0: print(f'images {i} of {n_batches}')
    scores.append(score_fn(X).item())
  scores = np.array(scores)
  return scores