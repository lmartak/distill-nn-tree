def brand_new_tfsession(sess=None):

    import tensorflow as tf
    from tensorflow.keras.backend import set_session

    if sess:
        tf.reset_default_graph()
        sess.close()

    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)

    return sess

def draw_tree(sess, tree, img_rows, img_cols, img_chans,
              input_img=None, show_correlation=False, savepath=''):

    import itertools
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import ConnectionPatch

    def _add_arrow(ax_parent, ax_child, xyA, xyB, color='black'):
        '''Private utility function for drawing arrows between two axes.'''
        con = ConnectionPatch(xyA=xyA, xyB=xyB, coordsA='data', coordsB='data',
                              axesA=ax_child, axesB=ax_parent, arrowstyle='<|-',
                              color=color)
        ax_child.add_artist(con)

    # collect model parameters for plotting
    kernels = dict([(l.name.split('_')[-1],
                     np.squeeze(l.get_weights()[0]).reshape(
                         (img_rows, img_cols, img_chans)))
                    for l in tree.model.layers if 'dense' in l.name])
    biases = dict([(l.name.split('_')[-1], np.squeeze(l.get_weights()[1][0]))
                   for l in tree.model.layers if 'dense' in l.name])
    leaves = dict([(l.name.split('_')[-1], sess.run(l.output))
                   for l in tree.model.layers if 'pdist' in l.name])

    n_leaves = 2**tree.max_depth
    assert len(leaves) == n_leaves

    # prepare figure and specify grid for subplots
    fig = plt.figure(figsize=(n_leaves, n_leaves//2))
    gs = GridSpec(tree.max_depth+1, n_leaves*2,
                  height_ratios=[1]*tree.max_depth+[0.5])

    # Grid Coordinate X (horizontal)
    gcx = [list(np.arange(1, 2**(i+1), 2) * (2**(tree.max_depth+1) // 2**(i+1)))
           for i in range(tree.max_depth+1)]
    gcx = list(itertools.chain.from_iterable(gcx))
    axes = {}
    path = ['0']

    imshow_args = {'origin': 'upper', 'interpolation': 'None', 'cmap': 'gray'}

    # draw tree nodes
    for pos, key in enumerate(sorted(kernels.keys(), key=lambda x:(len(x), x))):
        ax = plt.subplot(gs[len(key)-1, gcx[pos]-2:gcx[pos]+2])
        axes[key] = ax
        kernel_image = kernels[key]
        if input_img is not None and key in path:
            logit = sess.run(tree.inv_temp)[0] * (
                np.sum(input_img * kernels[key]) + biases[key])
            path.append(key + ('1' if (logit) >= 0 else '0'))
            ax.text(img_cols//2, img_rows+2, '{:.2f}'.format(logit),
                    ha='center', va='center')
            if show_correlation:
                kernel_image = input_img * kernels[key]
        ax.imshow(kernel_image.squeeze(), **imshow_args)
        ax.axis('off')
        digits = set([np.argmax(leaves[k]) for k in leaves.keys()
                      if k.startswith(key)])
        title = ','.join(str(digit) for digit in digits)
        plt.title('{}'.format(title))

    # draw tree leaves
    for pos, key in enumerate(sorted(leaves.keys(), key=lambda x:(len(x), x))):
        ax = plt.subplot(gs[len(key)-1,
                            gcx[len(kernels)+pos]-1:gcx[len(kernels)+pos]+1])
        axes[key] = ax
        leaf_image = np.ones((tree.n_classes, 1)) @ leaves[key]
        ax.imshow(leaf_image, **imshow_args)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title('{}'.format(np.argmax(leaves[key])), y=-.5)

    # add arrows indicating flow
    for pos, key in enumerate(sorted(axes.keys(), key=lambda x:(len(x), x))):
        children_keys = [k for k in axes.keys()
                         if len(k) == len(key) + 1 and k.startswith(key)]
        for child_key in children_keys:
            p_rows, p_cols = axes[key].get_images()[0].get_array().shape
            c_rows, c_cols = axes[child_key].get_images()[0].get_array().shape
            color = 'green' if (key in path and child_key in path) else 'red'
            _add_arrow(axes[key], axes[child_key],
                       (c_cols//2, 1), (p_cols//2, p_rows-1), color)


    # draw input image with arrow indicating flow into the root node
    if input_img is not None:
        ax = plt.subplot(gs[0, 0:4])
        ax.imshow(input_img.squeeze(), clim=(0.0, 1.0), **imshow_args)
        ax.axis('off')
        plt.title('input')
        _add_arrow(ax, axes['0'],
                   (1, img_rows//2), (img_cols-1, img_rows//2), 'green')

    if savepath:
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()

    return None
