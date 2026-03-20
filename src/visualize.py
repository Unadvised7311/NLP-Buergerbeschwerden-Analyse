import matplotlib.pyplot as plt

def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(1, 5, figsize=(20, 10), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, color='teal')
        ax.set_title(f"Thema {topic_idx + 1}")
        ax.invert_yaxis()
        for i in "top right left".split():
            ax.spines[i].set_visible(False)

    fig.suptitle(title, fontsize=20)
    return fig

def visualize_lda_placeholder():
    # Da pyLDAvis auf Arch/3.14 oft scheitert, erwähnen wir es als Option
    print("Interaktive pyLDAvis-Visualisierung vorbereitet.")
