# Introduction
Learning under supervision has had a profound impact in the progress of AI and machine learning over the past decades. Supervised learning, however, requires a large amount of data to achieve reasonable performance for a given task, which generally do not transfer effectively when learning a new set of tasks. In essence, to adapt to new tasks we would need to acquire new (large) data sets. Collecting and labelling the amount of data necessary to train supervised models is expensive and doesn’t scale well. For example, ImageNet with its 14 million images is estimated to take ~19 human years to annotate [[1][imgnet]], keeping in mind that the dataset: has limited concepts of the world; doesn’t include any temporal concepts; and is unbalanced and not fully inclusive [[2][bias1]]. In many real-world scenarios, we simply don’t have access to labels for the data. Medical imaging is notorious for this [[3][medGAN]] as professionals have to spend countless hours looking at images in order to manually classify and segment them, or machine translation for languages that aren’t as prevalent in digital form.

Self-supervised learning--where a model would generate the labels needed for learning from the data itself-- has made tremendous progress over the past few years, with successes in NLP [[4][bert]] and in video and language representation learning [[5][videobert]]. With self-supervision, the target is to significantly improve model performance even if only a fraction of the dataset is labeled.

In this project, we’ll implement a contrastive self-supervision model to learn image representations from unlabelled data and investigate its performance for an image classification downstream task. The overall outlined learning framework is semi-supervised and trained in two stages. First we’ll implement SimCLR to extract image feature representations without using labels. Then, we’ll use a small, labelled dataset to train a classifier on top of our learned feature backbone for the classification task.

# Related Work
The SimCLR paper written by Chen et al. [[6][simclr]] wholly lays out the foundation for the framework. The researchers involved have put forward a method that simplifies and improves upon previous approaches to contrastive learning on images. The paper concludes with some major findings that enable good contrastive learning which includes a comparison between different learning methods and how they behave when each component is modified in certain ways.

Another paper written by Le-Khac, Healy, and Smeaton [[7][creview]] explores the recent interest in contrastive learning in multiple domains, as well as the development and origins of it. The authors then provide a comprehensive literature review and propose another general framework that simplifies and unifies different contrastive learning methods. Finally, the paper walks through examples of how contrastive learning has been applied, as well as the challenges and promises that lie ahead

# Data
We plan to use ImageNet and/or CIFAR10/100. There is also potential in using the STL10 dataset as it contains both training and unlabeled splits for unsupervised and self-supervised learning tasks [[8][stl]].

# Methodology
The SimCLR model consists of a ResNet-50 feature backbone (with classification head removed) and a linear projection head g(). After training the contrastive model, we would discard the projection head and add an appropriate classification head for the downstream task.

The training procedure relies heavily on producing augmented pairs for each input datapoint. Those pairs, called positive samples, are the basis for the contrastive learning objective where we optimize our model to produce representations that achieve high similarity between the positive pairs and low similarity with all the other augmented data points (negatives). The paper presents multiple ablation studies on the impact of different augmentation procedures.

The most involved aspect of the project is implementing the contrastive loss function in a performant manner. Additionally, self-supervised training can require extended training times, which can limit our ability to experiment with different hyperparameters.

# Metrics
Our primary interest is evaluating how well a self-supervised model would perform on a classification task when compared to (1) a fully-supervised model trained on a small, labelled dataset, and (2) a supervised model trained on large amounts of data. In both settings, we are interested in the classification accuracy, but also on the trade-offs between accuracy and the size of the training data; the overall training time; and so on.

With these metrics in mind, our base goal is to reimplement SimCLR from the paper, and to benchmark its performance using pretrained weights in its backend. The next target is to fully train the SimCLR model from scratch. We plan on benchmarking the model on two experiments/setups:
#### Experiment A:
1. Train the SimCLR model on unlabelled data, discard the projection head and train a new classifier head on a small labelled dataset.
2. Train a _vanilla_ classifier on the same small labelled dataset.
3. Compare classification accuracy on a test set, track training loss and accuracy on each epoch.

#### Experiment B:
1. Use the same trained SimCLR model in the first experiment.
2. Train a traditional supervised classifier on a large labelled dataset.
3. Compare classification accuracy on a test set, track training loss and accuracy on each epoch.

For our stretch goals, we plan on performing more in-depth ablation studies on: the effect of different augmentation procedures on the learned representation, the impact of different batch sizes on training the contrastive model, and investigating points of convergence between the self-supervised and supervised models with respect to training data size or training time, for example. Finally we may test how well does the learned representation generalize across different domains, whether on different downstream tasks or in transfer learning. For that, we’ll test our model on other unseen datasets, as well as using our feature backend for other vision tasks such as object detection or image segmentation.

# Ethics
While the dataset(s) we eventually end up using (see Data) will be tailored for use in academic setting, current established biases in our choice of datasets [[9][bias2]] are expected to propagate to our model. More broadly, a problem inherent in the idea of training models using unlabeled data is that the end goal is the ability to train the model by leveraging the massive amounts of data available on the web, which exposes the model to the biases, such as racial and gender stereotypes, in the data.

The datasets we choose to use in our project also affect our choice of metrics to use for evaluating success. The primary metric of success we are using (see Metrics) is the classification accuracy of our model against a supervised model trained on a small, labeled dataset and a supervised model trained on a significantly larger dataset. It is crucial to note that this metric of success is only valid insofar as the aims of the original paper are concerned. For application of the SimCLR framework to domain-specific issues, additional metrics must be used. For example, in the case of facial recognition, one may want to compare the performance of the model on different skin tones to decrease the likelihood of racial bias being learned by the model.

# Division of Labor
Our tentative plan is to get everyone involved in each stage of the project, with one person taking a lead on one task. We anticipate that implementing the contrastive loss is the most challenging part (also where most of the meat is) so we’ll all contribute equally on this task.

Our tentative task breakdown is as follows:
- Preprocessing and data preparation,
- Implementing the simCLR model,
- Implementing the contrastive loss,
- Implementing the contrastive training procedure,
- Experimentation and ablation studies

# Referenced Implementations
- https://github.com/Spijkervet/SimCLR


[imgnet]: https://image-net.org/static_files/papers/ImageNet_2010.pdf
[bias1]: https://arxiv.org/abs/2010.15052
[medGAN]: https://arxiv.org/abs/1811.10669
[bert]: https://arxiv.org/abs/1810.04805
[videobert]: https://arxiv.org/abs/1904.01766
[simclr]: https://arxiv.org/abs/2002.05709
[creview]: https://arxiv.org/abs/2010.05113
[stl]: https://ai.stanford.edu/~acoates/stl10/
[bias2]: https://arxiv.org/abs/1911.11834
