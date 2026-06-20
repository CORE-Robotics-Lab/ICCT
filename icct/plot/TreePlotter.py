# adjusted from scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html


import numpy as np
from sklearn.tree._reingold_tilford import DrawTree, second_walk, third_walk, apportion, execute_shifts
from sklearn.tree._export import _color_brew
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.text import Annotation

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']


class BaseTreePlotter:
    def __init__(
            self,
            max_depth=None,
            feature_names=None,
            output_names=None,
            label="all",
            rounded=False,
            precision=2,
            fontsize=None,
    ):
        self.max_depth = max_depth
        self.feature_names = feature_names
        self.output_names = output_names
        self.label = label
        self.rounded = rounded
        self.precision = precision
        self.fontsize = fontsize
        self.paragraph_mode = False  # used to adjust the behavior for tree_node.get_str when generating paragraphs

    def get_color(self, value):
        # Find the appropriate color & intensity for a node
        if self.colors["bounds"] is None:
            # Classification tree
            color = list(self.colors["rgb"][np.argmax(value)])
            sorted_values = sorted(value, reverse=True)
            if len(sorted_values) == 1:
                alpha = 0
            else:
                alpha = (sorted_values[0] - sorted_values[1]) / (1 - sorted_values[1])
        else:
            # Regression tree or multi-output
            color = list(self.colors["rgb"][0])
            alpha = (value - self.colors["bounds"][0]) / (
                    self.colors["bounds"][1] - self.colors["bounds"][0]
            )
        # unpack numpy scalars
        alpha = float(alpha)
        # compute the color as alpha against white
        color = [int(round(alpha * c + (1 - alpha) * 255, 0)) for c in color]
        # Return html color code in #RRGGBB format
        return "#%2x%2x%2x" % tuple(color)

    def get_fill_color(self, tree, node_id):
        # Fetch appropriate color for node
        if "rgb" not in self.colors:
            # Initialize colors and bounds if required
            self.colors["rgb"] = _color_brew(tree.n_classes[0])
            if tree.n_outputs != 1:
                # Find max and min impurities for multi-output
                self.colors["bounds"] = (np.min(-tree.impurity), np.max(-tree.impurity))
            elif tree.n_classes[0] == 1 and len(np.unique(tree.value)) != 1:
                # Find max and min values in leaf nodes for regression
                self.colors["bounds"] = (np.min(tree.value), np.max(tree.value))
        if tree.n_outputs == 1:
            node_val = tree.value[node_id][0, :] / tree.weighted_n_node_samples[node_id]
            if tree.n_classes[0] == 1:
                # Regression
                node_val = tree.value[node_id][0, :]
        else:
            # If multi-output color node by impurity
            node_val = -tree.impurity[node_id]
        return self.get_color(node_val)

    def node_to_str(self, tree_node, node_id):
        return tree_node.get_str(precision=self.precision, feature_names=self.feature_names,
                                 paragraph_mode=self.paragraph_mode)


class TreePlotter(BaseTreePlotter):
    def __init__(
            self,
            max_depth=None,
            feature_names=None,
            output_names=None,
            label="all",
            rounded=False,
            precision=2,
            fontsize=None,
            highlighting=False
    ):

        super().__init__(
            max_depth=max_depth,
            feature_names=feature_names,
            output_names=output_names,
            label=label,
            rounded=rounded,
            precision=precision,
        )
        self.fontsize = fontsize
        self.highlighting = highlighting

        # The depth of each node for plotting with 'leaf' option
        self.ranks = {"leaves": []}
        # The colors to render each node with
        self.colors = {"bounds": None}

        self.characters = ["#", "[", "]", "<=", "\n", "", ""]
        self.bbox_args = dict()
        if self.rounded:
            self.bbox_args["boxstyle"] = "round"

        self.arrow_args = dict(arrowstyle="<-", linewidth=4.0)
        self._init_text_kwargs()

    def _init_text_kwargs(self):
        kwargs = dict(
            bbox=self.bbox_args.copy(),
            ha="center",
            va="center",
            zorder=100,
            xycoords="axes fraction",
            arrowprops=self.arrow_args.copy(),
        )

        if self.fontsize is not None:
            kwargs["fontsize"] = self.fontsize

        kwargs["bbox"]["fc"] = (1, 1, 1)
        kwargs["bbox"]["ec"] = (1, 1, 1)
        kwargs["bbox"]["linestyle"] = "--"
        kwargs["arrowprops"]["arrowstyle"] = "-"
        kwargs["arrowprops"]["linewidth"] = 0.0
        kwargs["bbox"]["linewidth"] = 4.0
        self.text_kwargs = kwargs

    def _make_tree(self, tree_node, depth=0):
        # traverses _tree.Tree recursively, builds intermediate
        # "_reingold_tilford.Tree" object
        label = self.node_to_str(tree_node, tree_node.node_id)
        if tree_node.children:
            children = [self._make_tree(child, depth=depth + 1) for child in tree_node.children]
            return LabeledTree(tree_node, label, tree_node.node_id, *children)
        else:
            return LabeledTree(tree_node, label, tree_node.node_id)

    def export_paragraph(self, decision_tree_root):
        self.paragraph_mode = True
        my_tree = self._make_tree(decision_tree_root)
        paragraph = []

        def traversal(node, stack, child_order):
            if node.children:
                for id, child in enumerate(node.children):
                    traversal(node.children[id], stack + [node], child_order + [id])
            else:
                description = r"\textbf{If }"
                stack = stack + [node]
                for i, one_node in enumerate(stack):
                    if one_node.original_node.is_leaf():
                        break
                    if i != 0:
                        description += r"\text{and }"
                    if child_order[i] == 0:
                        # evaluated True
                        description += one_node.label + r"\text{, }"
                    else:
                        # evaluated False
                        new_label = one_node.label
                        if r"\gt" in new_label:
                            new_label = new_label.replace(r"\gt", r"\lt")
                        else:
                            new_label = new_label.replace(r"\lt", r"\gt")
                        description += new_label + r"\text{, }"
                description += r"\\\quad\quad\textbf{Then, }"
                for j in range(i, len(stack)):
                    if self.output_names:
                        description += r"\text{action }%s=%s" % (self.output_names[j - i], stack[j].label)
                    else:
                        description += r"\text{action }%d=%s" % (j - i, stack[j].label)
                    if j < len(stack) - 1:
                        description += r"\text{, }"
                    else:
                        description += r"\text{. }\\"
                paragraph.append(description)

        traversal(my_tree, [], [])
        return paragraph

    def export(self, decision_tree_root, ax=None):
        self.paragraph_mode = False

        if ax is None:
            ax = plt.gca()
        ax.clear()
        ax.set_axis_off()
        my_tree = self._make_tree(decision_tree_root)
        draw_tree = buchheim(my_tree)

        if self.output_names is not None:
            # prepare locations for output_names
            output_name_locations = []
            left_to_left_most_leaf = 0.8
            current_node = draw_tree
            while True:
                if current_node.tree.original_node.is_leaf():
                    output_name_locations.append((current_node.x - left_to_left_most_leaf, current_node.y))
                if len(current_node.children) == 0:
                    break
                current_node = current_node.children[0]
            assert len(output_name_locations) == len(self.output_names)

        # important to make sure we're still
        # inside the axis after drawing the box
        # this makes sense because the width of a box
        # is about the same as the distance between boxes
        max_x, max_y = draw_tree.max_extents() + 1
        ax_width = ax.get_window_extent().width
        ax_height = ax.get_window_extent().height

        scale_x = ax_width / max_x
        scale_y = ax_height / max_y
        self.recurse(draw_tree, ax, max_x, max_y)
        if self.output_names is not None:
            # plot output_names
            for name, location in zip(self.output_names, output_name_locations):
                xy = ((location[0] + 0.5) / max_x, (max_y - location[1] - 0.5) / max_y)
                # offset things by .5 to center them in plot
                ax.annotate(name, xy, **self.text_kwargs)

        # plot yes and no on decision node branches
        def traversal(now_node, is_left_child):
            num_children = len(now_node.children)
            if num_children == 0:
                return
            if is_left_child is not None:
                text = "yes" if is_left_child else "no"
                xy = ((now_node.x + now_node.parent.x) / 2, (now_node.y + now_node.parent.y) / 2)
                xy = ((xy[0] + 0.5) / max_x, (max_y - xy[1] - 0.5) / max_y)
                ax.annotate(text, xy, **self.text_kwargs)
            if num_children > 1:
                traversal(now_node.children[0], True)
                traversal(now_node.children[1], False)
        traversal(draw_tree, is_left_child=None)

        anns = [ann for ann in ax.get_children() if isinstance(ann, Annotation)]

        # update sizes of all bboxes
        renderer = ax.figure.canvas.get_renderer()

        for ann in anns:
            ann.update_bbox_position_size(renderer)

        if self.fontsize is None:
            # get figure to data transform
            # adjust fontsize to avoid overlap
            # get max box width and height
            extents = [ann.get_bbox_patch().get_window_extent() for ann in anns]
            max_width = max([extent.width for extent in extents])
            max_height = max([extent.height for extent in extents])
            # width should be around scale_x in axis coordinates
            size = anns[0].get_fontsize() * min(
                scale_x / max_width, scale_y / max_height
            )
            for ann in anns:
                ann.set_fontsize(size)

        return anns

    def recurse(self, node, ax, max_x, max_y, depth=0):
        kwargs = dict(
            bbox=self.bbox_args.copy(),
            ha="center",
            va="center",
            zorder=100 - 10 * depth,
            xycoords="axes fraction",
            arrowprops=self.arrow_args.copy(),
        )
        kwargs["arrowprops"]["edgecolor"] = plt.rcParams["text.color"]

        if self.fontsize is not None:
            kwargs["fontsize"] = self.fontsize

        # offset things by .5 to center them in plot
        xy = ((node.x + 0.5) / max_x, (max_y - node.y - 0.5) / max_y)

        if self.max_depth is None or depth <= self.max_depth:
            # kwargs["bbox"]["fc"] = ax.get_facecolor()
            if node.tree.original_node.is_leaf():
                kwargs["bbox"]["fc"] = (241 / 255, 207 / 255, 205 / 255)
                kwargs["bbox"]["ec"] = (170 / 255, 87 / 255, 81 / 255)
                kwargs["bbox"]["linestyle"] = "--"
            else:
                kwargs["bbox"]["fc"] = (220 / 255, 232 / 255, 250 / 255)
                kwargs["bbox"]["ec"] = (107 / 255, 136 / 255, 184 / 255)
            if node.parent is not None and len(node.parent.children) == 1:
                # remove arrows with leave links
                kwargs["arrowprops"]["arrowstyle"] = "-"
                kwargs["arrowprops"]["linewidth"] = 0.0
            if self.highlighting:
                if node.tree.original_node.highlight:
                    kwargs["arrowprops"]["color"] = 'r'
                    kwargs["arrowprops"]["linewidth"] = 8.0
                    kwargs['fontweight'] = 'heavy'
                else:
                    kwargs['alpha'] = 0.5
                    kwargs["bbox"]['alpha'] = 0.5
                    kwargs["arrowprops"]['alpha'] = 0.5
            kwargs["bbox"]["linewidth"] = 4.0

            if node.parent is None:
                # root
                ax.annotate(node.tree.label, xy, **kwargs)
            else:
                xy_parent = (
                    (node.parent.x + 0.5) / max_x,
                    (max_y - node.parent.y - 0.5) / max_y,
                )
                ax.annotate(node.tree.label, xy_parent, xy, **kwargs)
            for child in node.children:
                self.recurse(child, ax, max_x, max_y, depth=depth + 1)

        else:
            xy_parent = (
                (node.parent.x + 0.5) / max_x,
                (max_y - node.parent.y - 0.5) / max_y,
            )
            kwargs["bbox"]["fc"] = "grey"
            ax.annotate("\n  (...)  \n", xy_parent, xy, **kwargs)


def recursive_highlight_parent(node):
    node.tree.original_node.highlight = True
    if node.parent is not None:
        recursive_highlight_parent(node.parent)


def recursive_highlight_children(node):
    node.tree.original_node.highlight = True
    for child in node.children:
        recursive_highlight_parent(child)


def first_walk(v, distance=1.0):
    if v.tree.original_node.highlight:
        recursive_highlight_parent(v)
        recursive_highlight_children(v)
    if len(v.children) == 0:
        if v.lmost_sibling:
            v.x = v.lbrother().x + distance
        else:
            v.x = 0.0
    else:
        default_ancestor = v.children[0]
        for w in v.children:
            first_walk(w)
            default_ancestor = apportion(w, default_ancestor, distance)
        # print("finished v =", v.tree, "children")
        execute_shifts(v)

        midpoint = (v.children[0].x + v.children[-1].x) / 2

        w = v.lbrother()
        if w:
            v.x = w.x + distance
            v.mod = v.x - midpoint
        else:
            v.x = midpoint
    return v


def second_walk(v, m=0, depth=0, leaf_depth=0, min=None):
    v.x += m
    if v.parent is not None and len(v.parent.children) == 1:
        v.y = depth + leaf_depth * 0.3
    else:
        v.y = depth

    if min is None or v.x < min:
        min = v.x

    for w in v.children:
        if v.tree.original_node.is_leaf():
            min = second_walk(w, m + v.mod, depth, leaf_depth + 1, min)
        else:
            min = second_walk(w, m + v.mod, depth + 1, leaf_depth, min)

    return min


def buchheim(tree):
    dt = first_walk(DrawTree(tree), distance=1.0)
    min = second_walk(dt)
    if min < 0:
        third_walk(dt, -min)
    return dt


class LabeledTree:
    def __init__(self, original_node, label="", node_id=-1, *children):
        self.original_node = original_node
        self.label = label
        self.node_id = node_id
        if children:
            self.children = children
        else:
            self.children = []


class NodeICCT:
    def __init__(self, node_id, highlight=False, *children):
        self.node_id = node_id
        self.highlight = highlight
        if children:
            self.children = children
        else:
            self.children = []

    def is_leaf(self):
        raise NotImplemented


class ManuallyLabelledNodeICCT(NodeICCT):
    def __init__(self, node_id, label, am_leaf, *children):
        self.label = label
        self.am_leaf = am_leaf
        super(ManuallyLabelledNodeICCT, self).__init__(node_id, *children)

    def get_str(self, **kwargs):
        if not kwargs["paragraph_mode"]:
            return self.label
        else:
            label = self.label.replace("$", "")
            return label

    def is_leaf(self):
        return self.am_leaf

class DecisionNodeICCT(NodeICCT):
    def __init__(self, weight, bias, alpha, node_id, *children):
        self.weight = weight
        self.bias = bias
        self.alpha = alpha
        super(DecisionNodeICCT, self).__init__(node_id, *children)

    def get_str(self, **kwargs):
        argmax_weight = np.argmax(np.abs(self.weight))
        if kwargs["paragraph_mode"]:
            gt_sign = r"\gt"
            lt_sign = r"\lt"

        else:
            gt_sign = ">"
            lt_sign = "<"

        if "feature_names" in kwargs.keys():
            feature_names = kwargs["feature_names"]
            feature_name = feature_names[argmax_weight]
            feature_name = r"\text{{{0:s}}}".format(feature_name)
        else:
            feature_name = "x^{{{0:d}}}".format(argmax_weight)

        node_string = r"{0:s}{1:s}{2:.{3}f}".format(feature_name,
                                                    gt_sign if self.weight[argmax_weight] > 0 else lt_sign,
                                                    self.bias / self.weight[argmax_weight],
                                                    kwargs["precision"])
        if not kwargs["paragraph_mode"]:
            node_string = "$" + node_string + "$"
        return node_string

    def is_leaf(self):
        return False


class DecisionLeafGaussianICCT(NodeICCT):
    def __init__(self, mu, std, node_id, *children):
        self.mu = mu
        self.std = std
        super(DecisionLeafGaussianICCT, self).__init__(node_id, *children)

    def get_str(self, **kwargs):
        node_string = r"\mathcal{{N}}({0:.{2}f}, {1:.{2}f})".format(self.mu,
                                                                    self.std,
                                                                    kwargs["precision"])
        if not kwargs["paragraph_mode"]:
            node_string = "$" + node_string + "$"
        return node_string

    def is_leaf(self):
        return True


class DecisionLeafSubmodelICCT(NodeICCT):
    def __init__(self, weight, bias, scalar, node_id, num_feature_to_show, *children):
        self.weight = weight
        self.bias = bias
        self.scalar = scalar
        self.num_feature_to_show = num_feature_to_show
        super(DecisionLeafSubmodelICCT, self).__init__(node_id, *children)

    def get_str(self, **kwargs):
        def get_feature_name(index):
            if "feature_names" in kwargs.keys():
                feature_names = kwargs["feature_names"]
                feature_name = feature_names[argmax_scalar[index]]
                feature_name = r" \times \text{{{0:s}}}".format(feature_name)
            else:
                feature_name = "x^{{{0:d}}}".format(argmax_scalar[index])
            return feature_name

        argmax_scalar = np.argpartition(np.abs(self.scalar), -self.num_feature_to_show)[-self.num_feature_to_show:]
        node_string = ""
        bias = 0
        # the first element does not need sign if it's +
        i = 0
        node_string += r"{1:.{2}f}{0:s}".format(get_feature_name(i),
                                                self.weight[argmax_scalar[i]] * self.scalar[argmax_scalar[i]],
                                                kwargs["precision"])

        bias += self.bias[argmax_scalar[i]] * self.scalar[argmax_scalar[i]]
        # starting the second, we need sign
        for i in range(1, self.num_feature_to_show):
            node_string += "{1:+.{2}f}x^{{{0:s}}}".format(get_feature_name(i),
                                                          self.weight[argmax_scalar[i]] * self.scalar[argmax_scalar[i]],
                                                          kwargs["precision"])
            bias += self.bias[argmax_scalar[i]] * self.scalar[argmax_scalar[i]]
        node_string += "{0:+.{1}f}".format(bias, kwargs["precision"])

        if not kwargs["paragraph_mode"]:
            node_string = "$" + node_string + "$"

        return node_string

    def is_leaf(self):
        return True


def plot_tree(
        decision_tree,
        *,
        max_depth=None,
        feature_names=None,
        output_names=None,
        label="all",
        rounded=False,
        precision=2,
        ax=None,
        fontsize=None,
        highlighting=False
):
    """Plot a decision tree.

    The sample counts that are shown are weighted with any sample_weights that
    might be present.

    The visualization is fit automatically to the size of the axis.
    Use the ``figsize`` or ``dpi`` arguments of ``plt.figure``  to control
    the size of the rendering.

    Read more in the :ref:`User Guide <tree>`.

    .. versionadded:: 0.21

    Parameters
    ----------
    decision_tree : decision tree regressor or classifier
        The decision tree to be plotted.

    max_depth : int, default=None
        The maximum depth of the representation. If None, the tree is fully
        generated.

    feature_names : list of strings, default=None
        Names of each of the features.
        If None, generic names will be used ("X[0]", "X[1]", ...).

    output_names : list of strings, default=None
        Names of each action.

    label : {'all', 'root', 'none'}, default='all'
        Whether to show informative labels for impurity, etc.
        Options include 'all' to show at every node, 'root' to show only at
        the top root node, or 'none' to not show at any node.

    rounded : bool, default=False
        When set to ``True``, draw node boxes with rounded corners and use
        Helvetica fonts instead of Times-Roman.

    precision : int, default=3
        Number of digits of precision for floating point in the values of
        impurity, threshold and value attributes of each node.

    ax : matplotlib axis, default=None
        Axes to plot to. If None, use current axis. Any previous content
        is cleared.

    fontsize : int, default=None
        Size of text font. If None, determined automatically to fit figure.

    highlighting : bool, default=False
        Whether in highlight mode

    Returns
    -------
    annotations : list of artists
        List containing the artists for the annotation boxes making up the
        tree.
    """

    exporter = TreePlotter(
        max_depth=max_depth,
        feature_names=feature_names,
        output_names=output_names,
        label=label,
        rounded=rounded,
        precision=precision,
        fontsize=fontsize,
        highlighting=highlighting
    )
    return exporter.export(decision_tree, ax=ax)


def tree_convert_to_text(
        decision_tree,
        *,
        max_depth=None,
        feature_names=None,
        output_names=None,
        label="all",
        rounded=False,
        precision=2,
        fontsize=None,
        highlighting=False
):
    """Plot a decision tree.

    The sample counts that are shown are weighted with any sample_weights that
    might be present.

    The visualization is fit automatically to the size of the axis.
    Use the ``figsize`` or ``dpi`` arguments of ``plt.figure``  to control
    the size of the rendering.

    Read more in the :ref:`User Guide <tree>`.

    .. versionadded:: 0.21

    Parameters
    ----------
    decision_tree : decision tree regressor or classifier
        The decision tree to be plotted.

    max_depth : int, default=None
        The maximum depth of the representation. If None, the tree is fully
        generated.

    feature_names : list of strings, default=None
        Names of each of the features.
        If None, generic names will be used ("X[0]", "X[1]", ...).

    output_names : list of strings, default=None
        Names of each action.

    label : {'all', 'root', 'none'}, default='all'
        Whether to show informative labels for impurity, etc.
        Options include 'all' to show at every node, 'root' to show only at
        the top root node, or 'none' to not show at any node.

    rounded : bool, default=False
        When set to ``True``, draw node boxes with rounded corners and use
        Helvetica fonts instead of Times-Roman.

    precision : int, default=3
        Number of digits of precision for floating point in the values of
        impurity, threshold and value attributes of each node.

    ax : matplotlib axis, default=None
        Axes to plot to. If None, use current axis. Any previous content
        is cleared.

    fontsize : int, default=None
        Size of text font. If None, determined automatically to fit figure.

    highlighting : bool, default=False
        Whether in highlight mode

    Returns
    -------
    annotations : list of artists
        List containing the artists for the annotation boxes making up the
        tree.
    """

    exporter = TreePlotter(
        max_depth=max_depth,
        feature_names=feature_names,
        output_names=output_names,
        label=label,
        rounded=rounded,
        precision=precision,
        fontsize=fontsize,
        highlighting=highlighting
    )
    return exporter.export_paragraph(decision_tree)


def build_icct_tree(model):
    all_variables = model.actor.ddt.state_dict()
    for key in all_variables.keys():
        all_variables[key] = all_variables[key].cpu().numpy()

    submodels = model.actor.ddt_kwargs["submodels"]
    num_decision_nodes = all_variables['layers'].shape[0]
    num_leaf_nodes = num_decision_nodes + 1
    if submodels:
        num_action_space = all_variables["sub_weights"].shape[1]
    else:
        num_action_space = all_variables["action_mus"].shape[1]

    # create all decision nodes
    nodes = []
    for i in range(num_decision_nodes):
        nodes.append(DecisionNodeICCT(weight=all_variables['layers'][i, :],
                                      bias=all_variables['comparators'][i, 0],
                                      alpha=all_variables['alpha'][0],
                                      node_id=i))
    # create all leaf nodes
    for j in range(num_action_space):
        for i in range(num_leaf_nodes):
            if submodels:
                nodes.append(DecisionLeafSubmodelICCT(weight=all_variables["sub_weights"][i, j, :],
                                                      bias=all_variables["sub_biases"][i, j, :],
                                                      scalar=all_variables["sub_scalars"][i, j, :],
                                                      node_id=num_decision_nodes + i,
                                                      num_feature_to_show=model.actor.ddt_kwargs["num_sub_features"]))
            else:
                nodes.append(DecisionLeafGaussianICCT(mu=all_variables['action_mus'][i, j],
                                                      std=all_variables['action_stds'][i, j],
                                                      node_id=num_decision_nodes + i))

    # connect decision nodes to decision nodes and decision nodes to first-layer leaves
    for i in range(num_decision_nodes):
        nodes[i].children = [nodes[i * 2 + 1], nodes[i * 2 + 2]]

    # connect leaves to leaves
    for j in range(num_action_space - 1):
        for i in range(num_leaf_nodes):
            nodes[num_decision_nodes + j * num_leaf_nodes + i].children = [
                nodes[num_decision_nodes + (j + 1) * num_leaf_nodes + i]]

    return nodes


def plot_icct_from_model(model, figsize, save_path, show=False, feature_names=None, output_names=None):
    nodes = build_icct_tree(model)
    fig, ax = plt.subplots(figsize=figsize)
    plot_tree(nodes[0], rounded=True, fontsize=30, ax=ax, feature_names=feature_names, output_names=output_names)
    plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()
    plt.cla()
    plt.clf()


def plot_icct_from_nodes(nodes, figsize, save_path, show=False, feature_names=None, output_names=None):
    fig, ax = plt.subplots(figsize=figsize)
    plot_tree(nodes[0], rounded=True, fontsize=30, ax=ax, feature_names=feature_names, output_names=output_names)
    plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()
    plt.cla()
    plt.clf()


def plot_highlight_path_icct(nodes, highlight_id, figsize, save_path, show=False, feature_names=None, output_names=None):
    nodes[highlight_id].highlight = True
    fig, ax = plt.subplots(figsize=figsize)
    plot_tree(nodes[0], rounded=True, fontsize=30, ax=ax, highlighting=True, feature_names=feature_names, output_names=output_names)
    plt.savefig(save_path)
    if show:
        plt.show()
    for node in nodes:
        node.highlight = False
    plt.close()
    plt.cla()
    plt.clf()


def pretty_observation_str(feature_names, feature_values, num_sets):
    """
    Make observation a pretty string, easier to view and understand

    :param feature_names: a list feature names
    :param feature_values: a list of feature values
    :param num_sets: how many rows should the generated string have
    :return: the formatted string
    """
    length = len(feature_names)
    assert length == len(feature_values)
    assert length % num_sets == 0
    num_items_each_set = int(length // num_sets)
    output_string = ""
    for i in range(num_sets):
        for j in range(num_items_each_set):
            item = i * num_items_each_set + j
            if j != 0:
                output_string += "\t%s: %.1f" % (feature_names[item], feature_values[item])
            else:
                output_string += "%s: %.2f" % (feature_names[item], feature_values[item])
        output_string += "\n"
    return output_string


def pretty_observation_html(feature_names, feature_values, num_sets):
    """
    Make observation a pretty string in HTML, easier to view and understand

    :param feature_names: a list feature names
    :param feature_values: a list of feature values
    :param num_sets: how many rows should the generated string have
    :return: the formatted string
    """
    length = len(feature_names)
    assert length == len(feature_values)
    assert length % num_sets == 0
    num_items_each_set = int(length // num_sets)
    output_string = """
        <table border="1" cellpadding="1" cellspacing="1" style="width:500px;">
         <thead>
          <tr>
           <th scope="col">&nbsp;</th>
           <th scope="col">Speed</th>
           <th scope="col">Location</th>
           <th scope="col">Lane</th>
          </tr>
         </thead>
        <tbody>
    """
    for i in range(num_sets):
        if i != num_sets - 1:
            output_string += """<tr>\n<td>Vehicle %d</td>\n""" % i
        else:
            output_string += """<tr>\n<td>Ego Vehicle</td>\n"""
        for j in range(num_items_each_set):
            item = i * num_items_each_set + j
            output_string += """<td>%.2f</td>\n""" % feature_values[item]
        output_string += """</tr>\n"""
    output_string += """</tbody>\n</table>\n"""
    return output_string


def pretty_observation_str_po_ring_lane_changing(feature_names, obs):
    for i, (name, ob) in enumerate(zip(feature_names, obs)):
        name = name.replace("\\", "")
        print("%d) %s: %.2f" % (i+1, name, ob))
