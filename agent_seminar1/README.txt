собачка пикси
кортесенькое овервью:

условие:
Pixel is a robot dog that learns new facts about the world and must decide
what to keep in memory and what to forget. The challenge: memory is limited
to only 10 items, but the stream of incoming facts can be unlimited.
1. Maximum 10 facts in memory at any time
2. Facts arrive one at a time from an incoming stream
3. Must answer questions by searching memory
4. When full, must decide which fact to evict
5. Accuracy should improve or remain stable over 30+ facts


что тут есть:
1. генератор фактов и вопросов:
     я решил не заморачиваться и поэтому факты имеют структуру "топик есть что-то", вопросы - "что есть топик?".\
     можно было бы усложнить эту систему (чтобы хотя бы грамматически было красиво), но я посчитал, что это лишнее
2. формат хранения фактов:
   - ID топика от 0 до 19 - (при больших количествах можно как-то хэшировать)
   - Age: этапов после того как записали
   - Query count: сколько раз мы им пользовались
   - Last access time: тут понятно
3. главная фишечка - сеть на обучении с подкреплением через q-обучение
   - Linear model: state_features -> Q-values per slot
   - Higher Q-value = more useful to KEEP
   - Adaptive learning rate
   - Trained via gradient descent
   - Stores (state, action, reward, next_state) tuples
   - Batch training from replay buffer
   - Helps with delayed reward credit assignment
   - Correct answer: +3.0 reward
   - Wrong answer (fact evicted): -5.0 penalty
   - Delayed credit assignment for evictions

пояснения к алгоритмам:
1. Q-LEARNING (for eviction decisions)
   - Q(s, a) = expected value of keeping fact in slot a
   - Update: Q(s,a) += alpha * (reward + gamma * max Q(s',a') - Q(s,a))
   - Uses linear function approximation
2. IMPORTANCE WEIGHTING (eviction bonus)
   - Query count: +1.0 per successful use
   - Topic frequency: +0.5 per time asked
   - Age penalty: -0.1 per step old
3. DELAYED REWARD CREDIT ASSIGNMENT
   - When a question fails, trace back to recent evictions (last 15)
   - Penalize the eviction decision that caused the failure
   - Helps learn which facts are actually important

претрейним on diverse data (15 seeds x 60 facts = 900 training samples), сохраняем q-функцию, дальше смотрим на\
то как отвечает на разных seed'ах.

сам код вызывается тут:
python pixel_memory.py

что заметили мы вместе с нейронкой(на самом деле чисто нейронка, я молча согласился):
1. Extended pre-training on diverse data is crucial for consistency
2. Importance weighting (query count + topic frequency) significantly helps
3. Accuracy degrades slowly (~6% per 10 extra topics(рассматривали увеличение топиков только до 40))
4. Variance is tight with trained weights (53-60%)

спасибо за такую возможность!!
