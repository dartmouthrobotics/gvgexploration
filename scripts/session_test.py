import random
from threading import Thread
SEND = 1
RECEIVE = 2


class BufferedData:
    def __init__(self, sid, msg):
        random.seed(1)
        self.sid = sid
        self.data = msg
        self.robot_id = None


class SessionManager:
    def __init__(self):
        self.name = 'session_manager'
        self.session_id = None
        self.sent_data = {}
        self.received_data = {}
        self.responses = {}
        self.ignored_received_data={}


    def handle_communication(self, command, data):
        if command == RECEIVE:
            thread = Thread(target=self.receive_data, args=(data))
            thread.start()
        else:
            thread = Thread(target=self.send_data, args=(data))
            thread.start()

    def receive_data(self, data):
        if not self.session_id:
            self.session_id = data.sid
            if data.robot_id not in self.received_data:
                self.received_data[data.robot_id] = [data]
            else:
                self.received_data[data.robot_id].append(data)
            data = BufferedData(random.randint(0, 1000), "message response from {}".format(self.robot_id))
            data.session_id = data.sessioin_id
            data.robot_id = self.robot_id
            self.handle_communication(SEND, data)
        else:
            if data.session_id == self.session_id:
                if self.session_id in self.responses:
                    self.responses[self.session_id].append(data.robot_id)
                else:
                    self.responses[self.session_id] = [self.session_id]
            else:
                if data.robot_id not in self.received_data:
                    self.ignored_received_data[data.robot_id] = [data]
                else:
                    self.ignored_received_data[data.robot_id].append(data)

    def send_data(self, data):
        if self.session_id:
           print("cant send again..already in session")
        else:
            data.session_id = data.sid
            self.handle_communication(data,SEND)

if __name__=='__main__':
    session_manager=SessionManager()
    while True:
        rid = random.randint(0,5)
        data = BufferedData(random.randint(0, 1000), "Initial message from {}".format(rid))
        data.robot_id=rid
        session_manager.send_data(data)






