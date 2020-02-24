# def start_gvg_exploration(self, goal):
#     success = True
#     # self.received_first = True
#     pose = (goal.pose.position.x, goal.pose.position.y, goal.pose.position.z)
#     edge = self.get_closest_leaf(pose)
#     if not edge:
#         self.exploration_result.result = "Invalid ridge"
#         self.action_server.set_succeeded(self.exploration_result)
#     else:
#         leaf = edge[1]
#         leaf_parent = edge[0]
#         self.create_feedback(leaf)
#         S = [leaf]
#         parents = {leaf: leaf_parent}
#         visited = {}
#         pixel = pose2pixel(pose, self.origin_x, self.origin_y, self.resolution)
#         aux_points = self.compute_aux_nodes(pixel, self.double_edges[edge])
#         aux_nodes = {leaf: aux_points}
#         leaf_edges = {leaf: edge}
#         # DFS starts here
#         while len(S) > 0:
#             if self.action_server.is_preempt_requested():
#                 rospy.loginfo('gvg_exploration: Preempted')
#                 self.action_server.set_preempted()
#                 success = False
#                 break
#
#             # ---------
#             u = S.pop()
#             # You can then decide on whether to head u or not
#             auxs = aux_nodes[u]
#             self.create_feedback(u)
#             self.action_server.publish_feedback(self.exploration_feedback)
#             unvisited_ans = [u]
#             for an in auxs:
#                 if not self.is_visited(an, visited):
#                     unvisited_ans.append(an)
#                     parents[an] = u
#             self.bfs(unvisited_ans, visited)
#             next_node = u
#             if next_node not in self.adj_list:
#                 pose = self.get_robot_pose()
#                 edge = self.get_closest_leaf(pose)
#                 if edge:
#                     next_node = edge[1]
#             if next_node:
#                 neighbors = self.adj_list[next_node]
#                 edge_dist, aux_ps = self.next_stop(next_node, neighbors, visited)
#                 # rospy.logerr("Robot {}: next aux next node: {}".format(self.robot_id, aux_ps))
#                 for v in neighbors:
#                     if v in aux_ps:
#                         aus_dict = aux_ps[v]
#                         edict = edge_dist[v]
#                         new_leaf = list(aus_dict.keys())[0]
#                         # if not self.is_visited(new_leaf, visited):
#                         S.append(new_leaf)
#                         parents[new_leaf] = next_node
#                         aux_nodes.update(aus_dict)
#                         leaf_edges[new_leaf] = edict[new_leaf]
#                     else:
#                         rospy.logerr('"Robot {}:Keeping the actual neighbor'.format(self.robot_id))
#                         S.append(v)
#                         parents[v] = next_node
#                 rospy.logerr("Robot {}: DFS next node: {}".format(self.robot_id, S))
#
#         if success:
#             self.create_result(visited)
#             rospy.loginfo('gvg exploration: Succeeded')
#             self.action_server.set_succeeded(self.exploration_result)





# def get_closest_ridge(self, pose):
#     close_edge = None
#     robot_pose = pose2pixel(pose, self.origin_x, self.origin_y, self.resolution)
#     closest_ridge = {}
#     edge_list = list(self.edges)
#     vertex_dict = {}
#     for e in edge_list:
#         p1 = e[0]
#         p2 = e[1]
#         o = self.edges[e]
#         width = D(o[0], o[1])
#         v1 = get_vector(p1, p2)
#         desc = (v1, width)
#         vertex_dict[e] = desc
#         d = min([D(robot_pose, e[0]), D(robot_pose, e[1])])
#         closest_ridge[e] = d
#     if closest_ridge:
#         close_edge = min(closest_ridge, key=closest_ridge.get)
#         # if closest_ridge[cr] < vertex_dict[cr][1]:
#         #     close_edge = cr
#     return close_edge, vertex_dict



def next_stop(self, u, neighbors, visited):
    leaf_neighbors = {}
    aux_ps = {}
    edge_dict = {}
    while not aux_ps:
        for n in neighbors:
            leaf_dict = self.get_child_leaf(n, u, visited)  # get end leaf, edge and corresponding obstacles
            leaf_neighbors[n] = leaf_dict

        for n, leaves in leaf_neighbors.items():
            new_edge_dict = {}
            new_aux_dict = {}
            for l_edge, edge_obs in leaves.items():
                new_leaf = l_edge[1]
                ans = self.compute_aux_nodes(new_leaf, edge_obs)
                new_aux_dict[new_leaf] = ans
                new_edge_dict[new_leaf] = l_edge
            aux_ps[n] = new_aux_dict
            edge_dict[n] = new_edge_dict

    return edge_dict, aux_ps