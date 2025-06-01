import os
import slackweb
# Slack Webhook URL
SLACKURL = 'YOUR_SLACK_WEBHOOK_URL_HERE'
# slack送信メソッド
def slackPost(message):
    slack = slackweb.Slack(url = SLACKURL)
    slack.notify(text = message)

if __name__ == '__main__':
    slackPost('fugu_finish')
    print('test')