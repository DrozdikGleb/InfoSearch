from csv import reader
import heapq
from collections import defaultdict
import functools


class Path:
    def __init__(self, prob, pos, node, edit_num):
        self.prob = prob
        self.pos = pos
        self.node = node
        self.edit_num = edit_num

    def __lt__(self, path2):
        return self.prob > path2.prob


class Node:
    def __init__(self):
        self.prob = 0.0
        self.children = {}
        self.query = ""


class Trie:
    def __init__(self):
        self.root = Node()
        self.def_prob = 0.001

    def add_word(self, word, prob):
        if len(word) > 15:
            return
        word = word + "$"
        self.add_word_inner(self.root, word, 0, prob)

    def add_word_inner(self, node, word, i, prob):
        node.prob = max(prob, node.prob)
        if i < len(word):
            elem = word[i]
            if elem in node.children:
                child = node.children[elem]
            else:
                child = Node()
                if elem == "$":
                    child.query = node.query
                else:
                    child.query = node.query + elem
                node.children[elem] = child
            self.add_word_inner(child, word, i + 1, prob)

    def save_top_k(self, k):
        self.save_top_k_inner(self.root, k)

    def save_top_k_inner(self, node, k):
        if len(node.children) == 0:
            return
        sort_children = sorted(node.children.items(), key=lambda x: x[1].prob, reverse=True)
        sort_children = sort_children[:min(len(sort_children), k)]
        node.children = dict((x, y) for x, y in sort_children)
        for key, value in node.children.items():
            self.save_top_k_inner(value, k)

    def get_candidates(self, word, lang_model, original_word_prob):
        word = word + "$"
        result = []
        queue = []
        heapq.heappush(queue, Path(1.0, 0, self.root, 0))
        while len(queue) > 0:
            path_prev = heapq.heappop(queue)
            prob_prev = path_prev.prob
            i = path_prev.pos
            node_prev = path_prev.node
            edit_num = path_prev.edit_num
            if i < len(word):
                error = word[i]
                if error not in node_prev.children and edit_num >= 1:
                    return result
                for correct, node_curr in node_prev.children.items():
                    cur_edit_num = edit_num
                    if correct == error:
                        prob_curr = lang_model[correct][error] if lang_model[correct][error] else 1.0
                    else:
                        prob_curr = lang_model[correct][error] if lang_model[correct][error] else self.def_prob
                        cur_edit_num += 1
                    prob_curr *= prob_prev
                    if cur_edit_num <= 1 and prob_curr * node_curr.prob > original_word_prob:
                        heapq.heappush(queue, Path(prob_curr, i + 1, node_curr, edit_num))
            elif len(node_prev.children) == 0:
                result.append(node_prev.query)
                if len(result) >= 5:
                    return result
        return result


def lev_distance(word1, word2):
    n = len(word1)
    m = len(word2)
    p = []
    dp = defaultdict(functools.partial(defaultdict, int))
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(dp[i - 1][j] + 1,
                           dp[i][j - 1] + 1,
                           dp[i - 1][j - 1] + (word1[i - 1] != word2[j - 1]))

    i = n
    j = m
    while i > 0 and j > 0:
        replace_or_match = dp[i][j] + 1 if word1[i - 1] != word2[j - 1] else dp[i][j]
        if replace_or_match <= dp[i][j - 1] + 1 and replace_or_match <= dp[i - 1][j] + 1:
            p.append(("R" if word1[i - 1] != word2[j - 1] else "M", word1[i - 1], word2[j - 1]))
            i -= 1
            j -= 1
        elif dp[i][j - 1] < dp[i - 1][j]:
            p.append(("I", '', word2[j - 1]))
            j -= 1
        else:
            p.append(("D", word1[i - 1], ''))
            i -= 1
    while i > 0:
        p.append(("D", word1[i - 1], ''))
        i -= 1
    while j > 0:
        p.append(("I", '', word2[j - 1]))
        j -= 1

    p.reverse()
    return dp[n][m], p


class ErrorModel:
    def __init__(self, train_file):
        self.kcg_model = defaultdict(functools.partial(defaultdict, float))
        self.known_res = {}
        self.parse_train(train_file)

    def parse_train(self, train_file):
        count = defaultdict(int)
        print("Start building error model")
        with open(train_file) as csv_file:
            csv_reader = reader(csv_file)
            next(csv_reader)
            for row in csv_reader:
                misspelled_word = row[0]
                correct_word = row[1]
                distance, prescription = lev_distance(misspelled_word, correct_word)
                for (op, x, w) in prescription:
                    if op == 'R' or op == 'M':
                        self.kcg_model[x][w] += 1
                for w in correct_word:
                    count[w] += 1
                self.known_res[misspelled_word] = correct_word

        for correct in self.kcg_model:
            for error in self.kcg_model[correct]:
                self.kcg_model[correct][error] /= count[correct]
        print("Finished building error model")


class LanguageModel:
    def __init__(self, vocab_file):
        self.unigram_model = defaultdict(float)
        self.parse_vocab(vocab_file)

    def parse_vocab(self, vocab_file):
        total = 0
        print("Start building language model")
        with open(vocab_file, "r") as word_freq_file:
            csv_reader = reader(word_freq_file)
            next(csv_reader)
            for line in csv_reader:
                self.unigram_model[line[0]] = int(line[1])
                total += int(line[1])
        for cur_word in self.unigram_model:
            self.unigram_model[cur_word] /= total
        print("Finished building language model")

    def get_word_prob(self, word):
        return self.unigram_model[word]


class SpellChecker:
    def __init__(self, trie, error_model, lang_model):
        self.trie = trie
        self.error_model = error_model
        self.lang_model = lang_model

    def write_to_output_file(self, no_fix_file_name, fix_file_name):
        print("Start word corrections")
        i = 0
        with open(fix_file_name, "w") as fix_file:
            with open(no_fix_file_name) as no_fix_file:
                csv_reader = reader(no_fix_file)
                next(csv_reader)
                fix_file.write("Id,Predicted\n")
                for line in csv_reader:
                    i += 1
                    if i % 5000 == 0:
                        print("Processed " + str(i) + " words")
                    word = line[0]
                    correct_word = self.correct_word(word)
                    fix_file.write(word + "," + correct_word + "\n")
        print("Finished word corrections")

    def correct_word(self, word):
        if word in self.error_model.known_res:
            return self.error_model.known_res[word]

        word_prob = self.lang_model.get_word_prob(word) * self.replace_prob(word, word)
        candidates = self.trie.get_candidates(word, self.error_model.kcg_model, word_prob)
        best_prob = word_prob
        best_word = word

        for candidate in candidates:
            cur_prob = self.lang_model.get_word_prob(candidate) * self.replace_prob(word,
                                                                                    candidate)
            if cur_prob > best_prob:
                best_prob = cur_prob
                best_word = candidate
        if best_word != word:
            print(word + " -> " + best_word)
        return best_word

    def replace_prob(self, word, correction):
        prob = 1.0
        distance, prescription = lev_distance(word, correction)
        if distance > 1:
            return 0.0
        for (op, cor, err) in prescription:
            if op == 'R' or op == 'M':
                prob *= self.error_model.kcg_model[cor][err]
        return prob


def main():
    lang_model = LanguageModel("words.csv")
    trie = Trie()
    for word, prob in lang_model.unigram_model.items():
        trie.add_word(word, prob)
    trie.save_top_k(5)
    error_model = ErrorModel("train.csv")

    spellchecker = SpellChecker(trie, error_model, lang_model)
    spellchecker.write_to_output_file("no_fix.submission.csv", "fix.submission.csv")


if __name__ == '__main__':
    main()
