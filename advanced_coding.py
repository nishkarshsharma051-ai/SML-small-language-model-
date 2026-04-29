"""
advanced_coding.py — Advanced coding questions and solutions for Ting Ling Ling.
Includes complex algorithms, system design, and optimization techniques.
"""

ADVANCED_CODING = {
    "dynamic programming": {
        "concept": "Dynamic Programming (DP) is a technique for solving complex problems by breaking them down into simpler subproblems. It is used when subproblems overlap and have optimal substructure.",
        "problems": [
            {
                "name": "Longest Common Subsequence",
                "problem": "Find the length of the longest subsequence present in both strings.",
                "solution": "```python\ndef lcs(X, Y):\n    m, n = len(X), len(Y)\n    dp = [[0] * (n + 1) for _ in range(m + 1)]\n    for i in range(1, m + 1):\n        for j in range(1, n + 1):\n            if X[i-1] == Y[j-1]:\n                dp[i][j] = dp[i-1][j-1] + 1\n            else:\n                dp[i][j] = max(dp[i-1][j], dp[i][j-1])\n    return dp[m][n]\n```"
            },
            {
                "name": "Knapsack Problem (0/1)",
                "problem": "Given weights and values of n items, put these items in a knapsack of capacity W to get the maximum total value.",
                "solution": "```python\ndef knapsack(W, wt, val, n):\n    dp = [[0] * (W + 1) for _ in range(n + 1)]\n    for i in range(1, n + 1):\n        for w in range(1, W + 1):\n            if wt[i-1] <= w:\n                dp[i][w] = max(val[i-1] + dp[i-1][w-wt[i-1]], dp[i-1][w])\n            else:\n                dp[i][w] = dp[i-1][w]\n    return dp[n][W]\n```"
            }
        ]
    },
    "graph algorithms": {
        "concept": "Graphs represent relationships between entities. Advanced algorithms involve shortest paths, spanning trees, and flow networks.",
        "algorithms": [
            {
                "name": "Dijkstra's Algorithm",
                "description": "Finds the shortest path from a source node to all other nodes in a weighted graph.",
                "implementation": "```python\nimport heapq\ndef dijkstra(graph, start):\n    distances = {node: float('inf') for node in graph}\n    distances[start] = 0\n    pq = [(0, start)]\n    while pq:\n        curr_dist, curr_node = heapq.heappop(pq)\n        if curr_dist > distances[curr_node]: continue\n        for neighbor, weight in graph[curr_node].items():\n            dist = curr_dist + weight\n            if dist < distances[neighbor]:\n                distances[neighbor] = dist\n                heapq.heappush(pq, (dist, neighbor))\n    return distances\n```"
            },
            {
                "name": "Kruskal's Algorithm",
                "description": "Finds the Minimum Spanning Tree (MST) of a graph using a Disjoint Set Union (DSU) data structure.",
                "implementation": "```python\ndef find(parent, i):\n    if parent[i] == i: return i\n    return find(parent, parent[i])\n\ndef union(parent, rank, x, y):\n    xroot, yroot = find(parent, x), find(parent, y)\n    if rank[xroot] < rank[yroot]: parent[xroot] = yroot\n    elif rank[xroot] > rank[yroot]: parent[yroot] = xroot\n    else: parent[yroot] = xroot; rank[xroot] += 1\n```"
            }
        ]
    },
    "system design": {
        "scalability": "Horizontal vs Vertical scaling. Horizontal adds more machines; Vertical adds more power (CPU/RAM) to one machine.",
        "load_balancing": "Distributes incoming network traffic across multiple servers. Algorithms: Round Robin, Least Connections, IP Hash.",
        "caching": "Stores copies of data in high-speed storage (Redis, Memcached) to reduce latency and database load. Strategies: Write-through, Write-back, LRU eviction.",
        "microservices": "Architectural style where an application is a collection of small, autonomous services modeled around a business domain."
    },
    "advanced data structures": {
        "trie": "A prefix tree used for efficient string searching and autocomplete. Complexity: O(L) where L is string length.",
        "segment_tree": "Allows range queries (sum, min, max) and point updates in O(log n) time.",
        "lru_cache": "Least Recently Used cache. Typically implemented with a Hash Map and a Doubly Linked List for O(1) access and update.",
        "fenwick_tree": "Also known as Binary Indexed Tree (BIT). Efficiently updates elements and calculates prefix sums in O(log n) time."
    },
    "system design patterns": {
        "microservices": "Architectural style where an application is a collection of small, autonomous services modeled around a business domain.",
        "event_driven": "Asynchronous communication between services using events. Tools: Kafka, RabbitMQ.",
        "database_sharding": "Partitioning data across multiple database instances to handle large volumes of data and traffic.",
        "consistency_patterns": "Strong consistency (linearizability), Eventual consistency (base), and Causal consistency."
    },
    "complexity analysis": {
        "time_complexity": "Describes the amount of time an algorithm takes to run as a function of the input size (n). Common: O(1), O(log n), O(n), O(n log n), O(n²).",
        "space_complexity": "Describes the amount of memory an algorithm uses relative to the input size.",
        "amortized_analysis": "Averages the time required to perform a sequence of operations over all operations performed."
    },
    "concurrency": {
        "python_gil": "The Global Interpreter Lock (GIL) prevents multiple native threads from executing Python bytecodes at once. Multiprocessing is used for CPU-bound tasks.",
        "asyncio": "Single-threaded, single-process design which uses cooperative multitasking. Ideal for I/O-bound tasks.",
        "deadlocks": "Occurs when two or more threads are blocked forever, each waiting for the other to release a resource."
    }
}
