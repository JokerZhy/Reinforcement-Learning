import gym

#print(gym.envs.registry.all())

env = gym.make('Ant-v2')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        print('observation:',observation)
        env.render()
        action = env.action_space.sample()
        '''
        action = 0 (for lest) or 1 (for right)
        '''
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
