def misra_gries(stream):
    k = 6
    counter = [0] * (k - 1)
    L = [None] * (k - 1)
    modL = 0
    print('Length of stream is :', len(stream))
    for i in range(0, len(stream)):
        if stream[i] in L:
            for j in range(0, len(L)):
                if stream[i] == L[j]:
                    counter[j] = counter[j] + 1
        else:
            for j in range(0, len(L)):
                if L[j] is None:
                    modL = 1
            if modL > 0:
                for j in range(0, len(L)):
                    if L[j] is None:
                        counter[j] = 1
                        L[j] = stream[i]
                        break
            else:
                for j in range(0, k - 1):
                    counter[j] = counter[j] - 1
        for j in range(0, k - 1):
            if counter[j] <= 0:
                L[j] = None
        modL = 0
    return L, counter
