import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

fig ,ax1 = plt.subplots()
y1 = [0.8112,0.7977,0.8,0.8012,0.7668,0.7412] # test_acc 折れ線グラフ
y2 = [14818.386,14080.14,13000,12882.975,9681.485,9059.194] # time  棒グラフ
t = [0.0,0.1,0.2,0.3,0.4,0.5] # threshold


c1, c2 = "blue", "red"  # 各プロットの色
l1, l2 = "test_acc", "training time" # 各ラベル

# 1つ目のグラフを描画
ax1.set_xlabel('threshold') #x軸ラベル
ax1.set_title("test_acc and training time") #グラフタイトル
ax1.grid() #罫線
ax1.plot(t, y1, color=c1, label=l1)
h1, l1 = ax1.get_legend_handles_labels()

# 2つ目のグラフを描画
ax2 = ax1.twinx()
ax2.bar(t, y2, color=c2, label=l2,alpha = 0.5, width = 0.05)
h2, l2 = ax2.get_legend_handles_labels()

ax1.legend(h1+h2, l1+l2, loc='upper right') # ax1とax2の凡例のhandlerとlabelのリストを結合
plt.show()

plt.savefig('./save/threshold_test_acc_time_el2n_1.png'.)


