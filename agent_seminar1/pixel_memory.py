import random
from collections import deque

MEMORY_SIZE = 10
TOPICS = [
    "food", "weather", "owner_preference", "school_subject", "name",
    "location", "color", "number", "animal", "time",
    "music", "book", "movie", "sport", "friend",
    "birthday", "address", "phone", "hobby", "pet"
]

def topic_to_id(topic):
    try:
        return TOPICS.index(topic)
    except ValueError:
        return len(TOPICS) - 1

def extract_topic(question):
    for topic in TOPICS:
        if topic in question.lower():
            return topic
    return "unknown"

class Fact:
    def __init__(self, topic, question, answer):
        self.topic = topic
        self.question = question
        self.answer = answer
        self.age = 0
        self.query_count = 0
        self.last_access_time = 0
        self.stored_at_step = 0
        self.times_evicted = 0

    def to_features(self):
        return [
            topic_to_id(self.topic) / len(TOPICS),
            min(self.age / 100.0, 1.0),
            min(self.query_count / 10.0, 1.0),
            min(self.last_access_time / 100.0, 1.0),
        ]

    def copy(self):
        f = Fact(self.topic, self.question, self.answer)
        f.age = self.age
        f.query_count = self.query_count
        f.last_access_time = self.last_access_time
        f.stored_at_step = self.stored_at_step
        f.times_evicted = self.times_evicted
        return f

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)

class QNetwork:
    def __init__(self, state_dim=40):
        self.state_dim = state_dim
        random.seed(42)
        self.weights = [[random.random() * 0.1 - 0.05 for _ in range(MEMORY_SIZE)] for _ in range(state_dim)]
        self.bias = [0.0] * MEMORY_SIZE
        self.lr = 0.05
        self.train_count = 0

    def forward(self, state):
        q_values = []
        for slot in range(MEMORY_SIZE):
            q = sum(state[i] * self.weights[i][slot] for i in range(len(state))) + self.bias[slot]
            q_values.append(q)
        return q_values

    def get_keep_values(self, state):
        return self.forward(state)

    def update(self, states, actions, targets, learning_rate=0.1):
        self.train_count += 1
        for s, a, t in zip(states, actions, targets):
            current_q = self.forward(s)[a]
            error = t - current_q
            for i in range(len(s)):
                self.weights[i][a] += learning_rate * error * s[i]
            self.bias[a] += learning_rate * error

class PixelMemory:
    def __init__(self):
        self.slots = [None] * MEMORY_SIZE
        self.current_step = 0
        self.empty_slots = list(range(MEMORY_SIZE))
        self.eviction_history = []
        self.q_network = QNetwork(state_dim=MEMORY_SIZE * 4)
        self.replay_buffer = ReplayBuffer()
        self.correct_answers = 0
        self.wrong_answers = 0
        self.total_questions = 0
        self.topic_ask_count = {topic: 0 for topic in TOPICS}
        self.topic_store_count = {topic: 0 for topic in TOPICS}

    def get_state(self):
        features = []
        for slot in self.slots:
            if slot is not None:
                features.extend(slot.to_features())
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
        return features

    def store(self, fact):
        fact.stored_at_step = self.current_step
        self.topic_store_count[fact.topic] += 1

        if self.empty_slots:
            slot_idx = self.empty_slots.pop(0)
            self.slots[slot_idx] = fact
            return slot_idx
        else:
            return self._evict_and_store(fact)

    def _evict_and_store(self, new_fact):
        state = self.get_state()
        keep_values = self.q_network.get_keep_values(state)

        for i, slot in enumerate(self.slots):
            if slot:
                keep_values[i] += slot.query_count * 1.0
                keep_values[i] += self.topic_ask_count.get(slot.topic, 0) * 0.5
                keep_values[i] -= slot.age * 0.1

        evict_slot = min(range(len(keep_values)), key=lambda i: keep_values[i])

        evicted_fact = self.slots[evict_slot]
        self.eviction_history.append({
            'slot': evict_slot,
            'fact': evicted_fact.copy() if evicted_fact else None,
            'step': self.current_step,
            'new_fact': new_fact,
            'reward': None
        })

        self.slots[evict_slot] = new_fact
        return evict_slot

    def answer_question(self, question):
        self.total_questions += 1
        question_topic = extract_topic(question)
        if question_topic in self.topic_ask_count:
            self.topic_ask_count[question_topic] += 1

        for slot, fact in enumerate(self.slots):
            if fact and fact.question == question:
                self.correct_answers += 1
                fact.query_count += 1
                fact.last_access_time = self.current_step
                self._provide_feedback(fact.question, correct=True)
                return True, fact.answer, fact

        self.wrong_answers += 1
        self._provide_feedback(question, correct=False)
        return False, None, None

    def _provide_feedback(self, question, correct):
        reward = 3.0 if correct else -5.0
        state = self.get_state()

        if correct:
            for slot, fact in enumerate(self.slots):
                if fact and fact.question == question:
                    self.replay_buffer.push(state.copy(), slot, reward, state.copy(), False)
                    break
        else:
            for eviction in self.eviction_history[-15:]:
                if eviction['reward'] is None:
                    eviction['reward'] = reward
                    evicted_fact = eviction.get('fact')
                    if evicted_fact and evicted_fact.question == question:
                        evicted_fact.times_evicted += 1

        if len(self.replay_buffer) >= 2:
            self._train_q_network()

    def _train_q_network(self):
        if len(self.replay_buffer) < 4:
            return

        batch = self.replay_buffer.sample(min(16, len(self.replay_buffer)))
        states = [b[0] for b in batch]
        actions = [b[1] for b in batch]
        targets = []

        for _, _, reward, next_state, done in batch:
            if done:
                targets.append(reward)
            else:
                future_q = max(self.q_network.forward(next_state)) if next_state else 0
                targets.append(reward + 0.9 * future_q)

        self.q_network.update(states, actions, targets, learning_rate=0.15)

    def age_facts(self):
        for fact in self.slots:
            if fact:
                fact.age += 1

class DatasetGenerator:
    def __init__(self, seed=42):
        random.seed(seed)
        self.answers = {
            "food": ["pizza", "sushi", "burger", "pasta", "salad", "ice cream", "tacos", "ramen", "steak", "soup"],
            "weather": ["sunny", "rainy", "cloudy", "snowy", "windy", "hot", "cold", "warm", "cool", "stormy"],
            "owner_preference": ["happy", "sad", "excited", "tired", "busy", "calm", "energetic", "relaxed", "creative", "curious"],
            "school_subject": ["math", "science", "history", "art", "music", "english", "geography", "biology", "physics", "chemistry"],
            "name": ["Max", "Bella", "Charlie", "Lucy", "Cooper", "Daisy", "Rocky", "Luna", "Duke", "Bailey"],
            "location": ["park", "school", "home", "store", "library", "zoo", "museum", "beach", "farm", "city"],
            "color": ["blue", "red", "green", "yellow", "purple", "orange", "pink", "black", "white", "brown"],
            "number": ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"],
            "animal": ["cat", "dog", "bird", "fish", "rabbit", "hamster", "turtle", "snake", "horse", "cow"],
            "time": ["morning", "afternoon", "evening", "night", "noon", "midday", "dawn", "dusk", "sunrise", "sunset"],
            "music": ["rock", "jazz", "pop", "classical", "hiphop", "blues", "country", "electronic", "reggae", "metal"],
            "book": ["novel", "poetry", "comic", "biography", "thriller", "fantasy", "history", "sci-fi", "mystery", "romance"],
            "movie": ["action", "comedy", "drama", "horror", "romance", "thriller", "documentary", "animation", "scifi", "fantasy"],
            "sport": ["soccer", "basketball", "tennis", "swimming", "running", "cycling", "golf", "volleyball", "baseball", "yoga"],
            "friend": ["Alex", "Sam", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Quinn", "Avery", "Blake"],
            "birthday": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October"],
            "address": ["123 Main St", "456 Oak Ave", "789 Pine Rd", "321 Elm Blvd", "654 Maple Dr", "987 Cedar Ln", "147 Birch Way", "258 Willow Ct", "369 Maple St", "741 Cedar Ave"],
            "phone": ["555-1234", "555-2345", "555-3456", "555-4567", "555-5678", "555-6789", "555-7890", "555-8901", "555-9012", "555-0123"],
            "hobby": ["painting", "photography", "gaming", "cooking", "gardening", "reading", "hiking", "fishing", "crafts", "dancing"],
            "pet": ["budgie", "parrot", "guinea pig", "ferret", "chinchilla", "hedgehog", "iguana", "frog", "gecko", "cockatiel"]
        }
        self.facts_generated = []

    def generate_fact(self, topic=None):
        if topic is None:
            topic = random.choice(TOPICS)
        subject = random.choice(self.answers[topic])
        question = f"what is {topic}"
        while any(f.question == question and f.answer == subject for f in self.facts_generated):
            subject = random.choice(self.answers[topic])
        fact = Fact(topic, question, subject)
        self.facts_generated.append(fact)
        return fact

    def generate_stream(self, num_facts, question_frequency=3):
        stream = []
        facts_pool = []
        for i in range(num_facts):
            if i % question_frequency == 0 and facts_pool:
                fact = random.choice(facts_pool)
                stream.append(('question', fact.question))
            else:
                fact = self.generate_fact()
                facts_pool.append(fact)
                stream.append(('fact', fact))
        return stream

class Pixel:
    def __init__(self, use_rl=True):
        self.memory = PixelMemory()
        self.use_rl = use_rl

    def learn(self, fact):
        self.memory.current_step += 1
        self.memory.store(fact)
        self.memory.age_facts()

    def query(self, question):
        correct, answer, _ = self.memory.answer_question(question)
        return correct, answer

def main():
    print("=" * 60)
    print("Pixel's Memory System - RL Eviction")
    print("=" * 60)
    print(f"Slots: {MEMORY_SIZE}, Topics: {len(TOPICS)}")
    print("=" * 60)

    print("Phase 1: Training (15 seeds x 60 facts)...")
    for warmup_seed in [42, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]:
        random.seed(warmup_seed)
        gen = DatasetGenerator(seed=warmup_seed)
        facts = [gen.generate_fact() for _ in range(60)]
        pixel = Pixel(use_rl=True)
        for f in facts:
            pixel.learn(f)
        for f in facts:
            pixel.query(f.question)

    saved_qnet = pixel.memory.q_network
    print(f"  Q-network trained: {saved_qnet.train_count} steps")

    print("\nPhase 2: Testing...")
    results = []
    for seed in [42, 10011, 20011, 30011, 40011]:
        random.seed(seed)
        gen = DatasetGenerator(seed=seed)
        facts = [gen.generate_fact() for _ in range(30)]

        pixel = Pixel(use_rl=True)
        pixel.memory.q_network = saved_qnet

        for f in facts:
            pixel.learn(f)
        for f in facts:
            pixel.query(f.question)

        correct = sum(1 for f in facts if pixel.query(f.question)[0])
        results.append(correct / len(facts))

    avg = sum(results) / len(results)
    print(f"RL Average: {avg:.0%}")
    print(f"Min: {min(results):.0%}, Max: {max(results):.0%}")
    print("=" * 60)

if __name__ == "__main__":
    main()