#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import codecs
import csv
import re
from enum import Enum


class Operation(Enum):
    AND = "AND",
    OR = "OR"


def build_index(docs, used_words):
    inverted_index = {}
    with open(docs) as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            doc_id = int(row[0])
            title = row[1]
            body = row[2]
            words = title.split() + body.split()
            for word in words:
                if word not in used_words:
                    continue
                postings = inverted_index.setdefault(word, list())
                if len(postings) > 0 and postings[-1] != doc_id or len(postings) == 0:
                    postings.append(doc_id)
    return inverted_index


def union(x, y):
    (i, j) = (0, 0)
    union = []
    while i < len(x) and j < len(y):
        if x[i] < y[j]:
            union.append(x[i])
            i = i + 1
        elif x[i] > y[j]:
            union.append(y[j])
            j = j + 1
        else:
            union.append(x[i])
            i = i + 1
            j = j + 1
    while i < len(x):
        union.append(x[i])
        i = i + 1
    while j < len(y):
        union.append(y[j])
        j = j + 1
    return union


def intersect(x, y):
    (i, j) = (0, 0)
    intersection = []
    while i < len(x) and j < len(y):
        if x[i] < y[j]:
            i += 1
        elif x[i] > y[j]:
            j += 1
        else:
            intersection.append(y[j])
            i += 1
            j += 1
    return intersection


class TreeNode:
    def __init__(self, data, operation: Operation = None):
        self.children = []
        self.operation = operation
        self.data = data

    def is_leaf(self):
        return len(self.children) == 0


class QueryTree:
    def __init__(self, query):
        self.search_res = []
        self.root = TreeNode(None, Operation.AND)
        self.used_words = set()
        self.build_and(self.root, query)

    def build_and(self, parent, query):
        inner_tokens = re.split(r"\s(?![^(]*\))", query)
        for cur_token in inner_tokens:
            if '|' in cur_token:
                new_node = TreeNode(None, Operation.OR)
                parent.children.append(new_node)
                self.build_or(new_node, self.remove_brackets(cur_token))
            else:
                parent.children.append(TreeNode(cur_token))
                self.used_words.add(cur_token)

    def build_or(self, parent, query):
        inner_tokens = query.split('|')
        for cur_token in inner_tokens:
            if ' ' in cur_token:
                new_node = TreeNode(None, Operation.AND)
                parent.children.append(new_node)
                self.build_and(new_node, cur_token)
            else:
                parent.children.append(TreeNode(cur_token))
                self.used_words.add(cur_token)

    def remove_brackets(self, query: str):
        return query.strip('()')

    def search(self, index):
        return self.search_inner(self.root, index)

    def search_inner(self, node, index):
        if node.is_leaf():
            return index.get(node.data, list())
        if node.operation is not None:
            op = intersect if node.operation == Operation.AND else union
            return self.apply_operation(op, node, index)

    def apply_operation(self, operation, node, index):
        children = node.children
        for child in children:
            postings = self.search_inner(child, index)
            if node.data is None:
                node.data = postings
            else:
                node.data = operation(node.data, postings)
        return node.data


class SearchResults:
    def __init__(self):
        self.res = {}

    def add(self, found, qid):
        self.res[qid] = found

    def print_submission(self, objects_file, submission_file):
        with open(submission_file, "w") as sub_file:
            with open(objects_file) as fd:
                for i, line in enumerate(fd, 0):
                    if i == 0:
                        sub_file.write("ObjectId,Relevance\n")
                        continue
                    row = line.split(',')
                    qid = int(row[1])
                    doc_id = int(row[2])
                    found = self.res[qid]
                    relevance = 1 if doc_id in found else 0
                    sub_file.write("%d,%d\n" % (i, relevance))


def main():
    # Command line arguments.
    parser = argparse.ArgumentParser(description='Homework 2: Boolean Search')
    parser.add_argument('--queries_file', required=True, help='queries.numerate.txt')
    parser.add_argument('--objects_file', required=True, help='objects.numerate.txt')
    parser.add_argument('--docs_file', required=True, help='docs.tsv')
    parser.add_argument('--submission_file', required=True, help='output file with relevances')
    args = parser.parse_args()
    csv.field_size_limit(1000000)

    search_results = SearchResults()
    query_trees = []
    used_words = set()
    with codecs.open(args.queries_file, mode='r', encoding='utf-8') as queries_fh:
        for line in queries_fh:
            fields = line.rstrip('\n').split('\t')
            qid = int(fields[0])
            query = fields[1]

            query_tree = QueryTree(query)
            used_words.update(query_tree.used_words)
            query_trees.append((qid, query_tree))

    index = build_index(args.docs_file, used_words)

    for elem in query_trees:
        qid, query_tree = elem
        res = query_tree.search(index)
        search_results.add(res, qid)

    search_results.print_submission(args.objects_file, args.submission_file)


if __name__ == "__main__":
    main()
