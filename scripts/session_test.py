import rospy
class SessionManager:
    def __init__(self):
        self.name = 'session_manager'
        # setting start time for:

        self.comm_session_time = 0   # from initiation of connection
        self.waiting_for_response = False # flag
        self.received_devices=[]  # list of robots which have responded

        self.auction_session_time = 0  # from initiation of auctioning
        self.waiting_for_auction_feedback = False  # flag

        self.time_after_bidding = 0    # starting time after bidding
        self.waiting_for_frontier_point = False # flag

        self.auction_waiting_time = 0  # start time after sending back data to sender
        self.waiting_for_auction = False # flag

        self.session_id = None # this is the key for the session
        self.is_sender=False # robot becomes sender if it has data to share

        self.max_wait_time = 60 # coming from params
        self.robot_id =0 # from param
        self.close_devices=[] # all devices that are within comm range

    def spin(self):
        while True:
            if not self.session_id:
                 self.check_data_sharing_status()
            else:
                time_to_go = self.evaluate_waiting()
                if time_to_go:
                    self.session_id = None  # you've  dropped the former commitment
                    self.resume_exploration()  # continue with your exploration

    def check_data_sharing_status(self):
        can_share=True
        if can_share:
            session_id = '{}_{}'.format(self.robot_id, rospy.Time.now().nsecs)
            self.session_id = session_id
            self.is_sender = True
            # send data
            self.comm_session_time = rospy.Time.now()
            self.waiting_for_response = True

    def evaluate_waiting(self):
        time_to_go = False
        if self.waiting_for_response:  # after sending data
            if (rospy.Time.now().to_sec() - self.comm_session_time) > self.max_wait_time:
                # just work with whoever has replied
                pass

        if self.waiting_for_frontier_point:  # after advertising an auction
            if (rospy.Time.now().to_sec() - self.time_after_bidding) > self.max_wait_time:
                self.waiting_for_frontier_point = False
                self.waiting_for_frontier_point = rospy.Time.now().to_sec()
                time_to_go = True

        if self.waiting_for_auction_feedback:  # after bidding in an auction
            if (rospy.Time.now().to_sec() - self.auction_session_time) > self.max_wait_time:
                self.waiting_for_auction_feedback = False
                self.auction_session_time = rospy.Time.now().to_sec(0)
                time_to_go = True

        if self.waiting_for_auction:  # after sending back data
            if (rospy.Time.now().to_sec() - self.auction_waiting_time) > self.max_wait_time:
                self.waiting_for_auction = False
                self.auction_waiting_time = rospy.Time.now().to_sec()
                time_to_go = True
        return time_to_go

    def request_and_share_frontiers(self):
        frontier_points =[] # request points from server
        if frontier_points:
            #publish frontier points

            # set auction waiting flags
            self.auction_session_time = rospy.Time.now().to_sec()
            self.waiting_for_auction_feedback = True
        else:
            self.received_devices = []
            self.is_sender = False  # reset this: you're not a sender anymore
            self.resume_exploration()  # say nothing, just go your way

        # reset the waiting flags, say nothing and move
        self.waiting_for_response = False
        self.comm_session_time = rospy.Time.now().to_sec()

    def resume_exploration(self):
        # resume exploration
        pass

    def on_receive_data(self,buff_data):
        sender_id = buff_data.sender_id
        if not buff_data.session_id:  # only respond to data with session id
            if not self.session_id:  # send back data and stop to wait for further communication
                self.session_id = buff_data.session_id
                self.push_messages_to_receiver([], self.session_id)
                self.cancel_exploration()

                # set auction waiting flags
                self.waiting_for_auction = True
                self.auction_waiting_time = rospy.Time.now().to_sec()

            else:  # So you have a session id
                if buff_data.session_id == self.session_id:
                    if self.is_sender:  # do this if you're a sender
                        self.received_devices.append(sender_id)
                        if len(self.received_devices) == len(self.close_devices):
                            self.request_and_share_frontiers()
                        else:  # do this if you're in a receive session
                            self.push_messages_to_receiver([sender_id],
                                                           None)  # just send back data but don't join the session
                else:  # do this if you're handling a send session
                    self.push_messages_to_receiver([sender_id], None)
        else:
            rospy.logerr('Robot {}: Received an tagged message'.format(self.robot_id))

    def push_messages_to_receiver(self,receivers,session_id):
        pass

    def cancel_exploration(self):
        pass

    def allocated_point_callback(self, data):
        # reset waiting for frontier point flags
        self.waiting_for_frontier_point = False
        self.time_after_bidding = rospy.Time.now().to_sec()
        # ============ End here =================
        # start exploration

    def auction_feedback_callback(self, data):
        if self.session_id:
            all_feedbacks = [data]
            if len(all_feedbacks) >= len(self.received_devices):
               # compute and share locations

                # reset waiting for bids flats and stop being a sender
                self.waiting_for_auction_feedback = False
                self.auction_session_time = rospy.Time.now().to_sec()
                self.is_sender = False
                # =============Ends here ==============
                # resume exploration
                all_feedbacks.clear()
                self.received_devices = []

    def auction_points_callback(self, data):
        if not self.is_sender:  # only participate if you're not a sender
            # process the auction and send it back

            # start waiting for location after bidding
            self.time_after_bidding = rospy.Time.now().to_sec()
            self.waiting_for_frontier_point = True

            # reset waiting for auction flags
            self.waiting_for_auction = False
            self.auction_waiting_time = rospy.Time.now().to_sec()



