    epsilon = TRAINING_EP if for_training else TESTING_EP

    epi_reward = 0  # None
    # initialize for each episode
    # TODO Your code here

    (current_room_desc, current_quest_desc, terminal) = framework.newGame()

    t = 0
    while not terminal:

        current_state_1 = dict_room_desc[current_room_desc]
        current_state_2 = dict_quest_desc[current_quest_desc]

        # Choose next action and execute
        # TODO Your code here
        if for_training:
            # update Q-function.
            # TODO Your code here
            ai, oi = epsilon_greedy(
                    state_1=current_state_1, 
                    state_2=current_state_2, 
                    q_func=q_func, 
                    epsilon=TRAINING_EP)

            next_room_desc, next_quest_desc, reward, terminal = framework.step_game(
                    current_room_desc,
                    current_quest_desc,
                    ai,
                    oi)

            # epi_reward += (GAMMA**t) * reward

            next_state_1 = dict_room_desc[next_room_desc]
            next_state_2 = dict_quest_desc[next_quest_desc]

            tabular_q_learning(q_func, 
                    current_state_1=current_state_1, 
                    current_state_2=current_state_2, 
                    action_index=ai,
                    object_index=oi, 
                    reward=epi_reward, 
                    next_state_1=next_state_1, 
                    next_state_2=next_state_2,
                    terminal=terminal)


        if not for_training:
            # update reward
            # TODO Your code here
            ai, oi = epsilon_greedy(
                    state_1=current_state_1, 
                    state_2=current_state_2, 
                    q_func=q_func, 
                    epsilon=TESTING_EP)

            next_room_desc, next_quest_desc, reward, terminal = framework.step_game(
                    current_room_desc,
                    current_quest_desc,
                    ai,
                    oi)

            epi_reward += (GAMMA**t) * reward


        # prepare next step
        # TODO Your code here
        current_room_desc = next_room_desc
        current_quest_desc = next_quest_desc

        t += 1


    if not for_training:
        return epi_reward



------------------------------------------------------------------------------------------------------------


    epsilon = TRAINING_EP if for_training else TESTING_EP

    epi_reward = 0  # None
    # initialize for each episode
    # TODO Your code here

    (current_room_desc, current_quest_desc, terminal) = framework.newGame()

    t = 0
    while not terminal:

        current_state_1 = dict_room_desc[current_room_desc]
        current_state_2 = dict_quest_desc[current_quest_desc]

        # Choose next action and execute
        # TODO Your code here
        ai, oi = epsilon_greedy(
                state_1=current_state_1,
                state_2=current_state_2,
                q_func=q_func,
                epsilon=epsilon)

        next_room_desc, next_quest_desc, reward, terminal = framework.step_game(
                current_room_desc,
                current_quest_desc,
                ai,
                oi)

        next_state_1 = dict_room_desc[next_room_desc]
        next_state_2 = dict_quest_desc[next_quest_desc]


        if for_training:
            # update Q-function.
            # TODO Your code here
            tabular_q_learning(q_func,
                    current_state_1=current_state_1,
                    current_state_2=current_state_2,
                    action_index=ai,
                    object_index=oi,
                    reward=epi_reward,
                    next_state_1=next_state_1,
                    next_state_2=next_state_2,
                    terminal=terminal)


        if not for_training:
            # update reward
            # TODO Your code here
            epi_reward += (GAMMA**t) * reward


        # prepare next step
        # TODO Your code here
        current_room_desc = next_room_desc
        current_quest_desc = next_quest_desc

        t += 1


    if not for_training:
        return epi_reward


