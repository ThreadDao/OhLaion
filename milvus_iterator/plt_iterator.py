import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    a = np.loadtxt("dir/iterator/search_iterator_cost_ms.txt")
    cost = a[:150]
    print(cost.shape)
    print(cost[0])
    print(cost[-1])
    print(np.average(cost))
    print(np.mean(cost))
    print(np.percentile(cost, 50))
    print(np.percentile(cost, 95))
    print(np.percentile(cost, 99))
    count = [i for i in range(cost.shape[0])]
    print(len(count))

    total_cost = np.sum(cost)
    print(f"total cost: {total_cost} ms")
    count = [i for i in range(len(cost))]

    cost_sort = np.sort(cost)
    print(cost_sort[0])
    print(cost_sort[-1])

    fig = plt.figure(figsize=(8, 6))
    plt.plot(count, cost)
    plt.title("search iteration cost")
    plt.xlabel("x-th iterator", size=16, color='black')
    plt.ylabel("iterator cost (s)", size=16, color='black')
    plt.legend(loc='upper left')
    plt.show()
    # # plt.savefig("it_limit_all.png")
