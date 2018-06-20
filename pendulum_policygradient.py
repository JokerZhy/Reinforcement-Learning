import gym
import tensorflow as tf
import numpy as np

EPISODES = 10000
BATCH_SIZE = 25
#H = 75 #failure
#H = 60 #550episode
#H = 32 #450
H=50
#learning_rate = 8e-2#425episode
learning_rate = 1e-1#250episode

D_state = 4
gamma = 0.99#discount factor

#建立单级倒立摆模型
env = gym.make('CartPole-v1')

'''
单隐层神经网络,无偏置;
隐层节点数：H;
隐层激活函数：ReLU,输出层激活函数：sigmoid
'''
input = tf.placeholder(tf.float32,[None,D_state])
W1 = tf.get_variable("W1",shape=[D_state,H], initializer=tf.contrib.layers.xavier_initializer())
#不使用tf.Variable创建的原因是，本例中的W还需要进行其它的
layer1 = tf.nn.relu(tf.matmul(input,W1))

W2 = tf.get_variable("W2",shape=[H,1],initializer=tf.contrib.layers.xavier_initializer())
output = tf.nn.sigmoid(tf.matmul(layer1,W2))#选择动作“1”的概率

train = tf.train.AdamOptimizer(learning_rate=learning_rate)
W1Grad = tf.placeholder(tf.float32, name='batch_grad1')
W2Grad = tf.placeholder(tf.float32, name='batch_grad2')
batchGrad = [W1Grad,W2Grad]

tvars = tf.trainable_variables()#获取所有可训练的变量
updateGrads = train.apply_gradients(zip(batchGrad, tvars))

#计算累积奖励
def discount_reward(r):
    discount_r = np.zeros_like(r)
    running_add = 0
    for i in reversed(range(r.size)):
        running_add = running_add * gamma + r[i]
        discount_r[i] = running_add
    return discount_r

#构造神经网络的“标签”和损失函数
label = tf.placeholder(tf.float32, [None, 1],name='label')
advantages = tf.placeholder(tf.float32, name='reward_signal')
loglik = tf.log(label * (label - output) + (1 - label) * (label + output))#????
loss = -tf.reduce_mean(loglik * advantages)

#tvars = tf.trainable_variables()#获取所有可训练的变量
newGrads = tf.gradients(loss , tvars)#计算loss相对于W1和W2的梯度

#初始化训练中需要用到的中间变量
xs, ys, drs = [],[],[]
reward_sum = 0
episode_count = 1 
with tf.Session() as sess:
    render_flag = False #图形渲染开关，为了减少延迟导致的程序运行时间，可在训练初始阶段关闭
    sess.run(tf.global_variables_initializer())
    observation = env.reset()
    gradBuffer = sess.run(tvars)#构造梯度缓冲器，每训练完一个BATCH_SIZE
    for ix, grads in enumerate(gradBuffer):
        gradBuffer[ix] = grads * 0#将初始的梯度置零
    while episode_count <= EPISODES:
        #if reward_sum / BATCH_SIZE > 100 or render_flag == True:
        #    env.render()
        #    render_flag = True
        state = np.reshape(observation,[1,D_state])
        tfprob = sess.run(output,feed_dict={input:state})
        action = 1 if np.random.uniform() < tfprob else 0
        #print('action:',action)
        xs.append(state)
        y = 1 - action
        ys.append(y)
          
        observation, reward, done, _ = env.step(action)
        reward_sum += reward
        #print('sum reward:',reward_sum)
        drs.append(reward)
        if done:
            episode_count +=1 
            epx = np.vstack(xs)#一个episode所有的状态
            epy = np.vstack(ys)#一个episode所有的label
            epr = np.vstack(drs)#一个episode所有的状态
            xs, ys, drs = [],[],[]#清空用于存储中间变量的列表
            discount_epr = discount_reward(epr)#计算累积奖励，并且标准化
            discount_epr = (discount_epr - np.mean(discount_epr)) / np.std(discount_epr)
            tGrad = sess.run(newGrads, feed_dict={input:epx,label:epy, advantages:discount_epr})
            for ix, grad in enumerate(tGrad):
                gradBuffer[ix] += grad#手动更新梯度缓冲器（相当于训练，但是神经网络未更新）
            if episode_count % BATCH_SIZE == 0:
                sess.run(updateGrads, feed_dict = {W1Grad:gradBuffer[0],W2Grad:gradBuffer[1]})#每训练BATCH_SIZE次，更新一次神经网络参数
                for ix, grads in enumerate(gradBuffer):
                    gradBuffer[ix] = grads * 0#将初始的梯度置零
                print("Averge reward for episode %d :%f." % (episode_count,reward_sum / BATCH_SIZE))
                if reward_sum / BATCH_SIZE >= 200:
                    print('Task solved in {} episode!'.format(episode_count))
                    break
                reward_sum = 0
            observation = env.reset()
