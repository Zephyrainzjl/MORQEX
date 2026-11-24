
#-------------------------------------------------------------------------------------#
# 图解析函数
#-------------------------------------------------------------------------------------#



def dfs(node, graph, visited, stack):
    visited[node] = True

    for neighbor in graph[node]:
        # if neighbor not in visited:  #neighbor可能不属于输入节点(只作为源,不作为目标)
        #     continue  #查询下一个节点 跳过该分支
        if not visited[neighbor]:   #尚未访问的节点 加入栈
            dfs(neighbor, graph, visited, stack)

    stack.append(node)

def transpose(graph):
    transposed_graph = {node: [] for node in graph}

    for node in graph:
        for neighbor in graph[node]:
            # if neighbor not in transposed_graph:
            #     continue  #查询下一个节点 该邻接不考虑
            transposed_graph[neighbor].append(node)

    return transposed_graph

def get_sccs(graph):
    visited = {node: False for node in graph}
    stack = []

    for node in graph:
        if not visited[node]:
            dfs(node, graph, visited, stack)

    transposed_graph = transpose(graph)
    visited = {node: False for node in graph}
    scc = []

    while stack:
        node = stack.pop()

        if not visited[node]:
            component = []
            dfs(node, transposed_graph, visited, component)
            scc.append(tuple(component)) 

    return scc



