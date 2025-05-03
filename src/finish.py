import os
import slackweb
# Slack Webhook URL
SLACKURL = 'https://hooks.slack.com/services/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'
# slack送信メソッド
def slackPost(message):
    slack = slackweb.Slack(url = SLACKURL)
    slack.notify(text = message)

if __name__ == '__main__':
    slackPost('fugu_finish')
    print('test')