import requests


class Bot:
    def __init__(self, token):
        self.token = token
        self.name = self.get_me()["first_name"]
        self.last_checked_update_id = 0

    def telegram_request(self, method, parameters=None, files=None):
        try:
            response = requests.post("https://api.telegram.org/bot" + self.token + "/" + method,
                                     params=parameters, files=files).json()
        except:
            response = {"ok": False}
        if response["ok"]:
            return response["result"]
        else:
            return {}

    def get_me(self):
        return self.telegram_request("getme")

    def get_updates(self, offset=None, limit=100, timeout=0, allowed_updates=None):
        params = {
            "offset": offset,
            "limit": limit,
            "timeout": timeout,
            "allowed_updates": allowed_updates
        }
        updates = self.telegram_request("getupdates", params)
        if len(updates) != 0:
            self.last_checked_update_id = updates[len(updates) - 1]["update_id"]
        return updates

    def get_file(self, file_id):
        params = {
            'file_id': file_id
        }
        return self.telegram_request('getfile', params)

    def send_message(self, chat_id, text):
        params = {
            'chat_id': chat_id,
            'text': text
        }
        return self.telegram_request("sendmessage", params)

    def get_last_messages(self):
        updates = self.get_updates(self.last_checked_update_id + 1, allowed_updates=["message"])
        messages = []
        for update in updates:
            if "message" in update.keys():
                messages.append(update["message"])
        return messages
