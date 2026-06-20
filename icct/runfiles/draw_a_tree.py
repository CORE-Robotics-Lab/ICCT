from icct.plot.TreePlotter import plot_icct_from_nodes, build_icct_tree, plot_highlight_path_icct, tree_convert_to_text, pretty_observation_str, pretty_observation_html, ManuallyLabelledNodeICCT


tree_figsize = (30, 10)
num_decision_nodes = 3
num_action_space = 2
num_leaf_nodes = 4
output_names = [r"grade\_math: ", r"grade\_essay: "]

nodes = []
nodes.append(ManuallyLabelledNodeICCT(0,
                                      r"$\text{hours\_game} \lt 2$",
                                      am_leaf=False))
nodes.append(ManuallyLabelledNodeICCT(1,
                                      r"$\text{hours\_homework} \gt 4$",
                                      am_leaf=False))
nodes.append(ManuallyLabelledNodeICCT(2,
                                      r"$\text{hours\_read} \gt 3$",
                                      am_leaf=False))
nodes.append(ManuallyLabelledNodeICCT(3,
                                      r"$10\times \text{hours\_homework} + 50$",
                                      am_leaf=True))
nodes.append(ManuallyLabelledNodeICCT(4,
                                      r"$5\times$ \text{hours\_homework} + 60",
                                      am_leaf=True))
nodes.append(ManuallyLabelledNodeICCT(5,
                                      r"$5\times$ \text{hours\_homework} + 55",
                                      am_leaf=True))
nodes.append(ManuallyLabelledNodeICCT(6,
                                      r"$-10\times$ \text{hours\_game} + 60",
                                      am_leaf=True))
nodes.append(ManuallyLabelledNodeICCT(7,
                                      r"$5\times$ \text{hours\_read} + 50",
                                      am_leaf=True))
nodes.append(ManuallyLabelledNodeICCT(8,
                                      r"$5\times$ \text{hours\_read} + 60",
                                      am_leaf=True))
nodes.append(ManuallyLabelledNodeICCT(9,
                                      r"$10\times$ \text{hours\_read} + 55",
                                      am_leaf=True))
nodes.append(ManuallyLabelledNodeICCT(10,
                                      r"$-10\times$ \text{hours\_game} + 60",
                                      am_leaf=True))

for i in range(num_decision_nodes):
    nodes[i].children = [nodes[i * 2 + 1], nodes[i * 2 + 2]]

# connect leaves to leaves
for j in range(num_action_space - 1):
    for i in range(num_leaf_nodes):
        nodes[num_decision_nodes + j * num_leaf_nodes + i].children = [
            nodes[num_decision_nodes + (j + 1) * num_leaf_nodes + i]]

text = tree_convert_to_text(nodes[0], rounded=True, fontsize=30,
                            output_names=output_names)
for line in text:
    print(r'<span class="katex">', line, r'</span>')
    print(r"</br>")
plot_icct_from_nodes(nodes, figsize=tree_figsize, save_path="temp_images/icct.png",
                     output_names=output_names)
