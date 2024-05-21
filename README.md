# VGAE-D

The method is based on variational graph autoencoder(VGAE) and it is used for graph level anomaly detection.

## Abstract

Graph anomaly detection plays a crucial role in identifying abnormal patterns within complex data structures, finding applications in various domains such as malicious molecule identification, financial fraud detection, and social network analysis. While existing research in graph anomaly detection predominantly focuses on node-level anomaly detection, there is a scarcity of methods specifically designed for graph-level anomaly detection. Moreover, these methods often fail to adequately explore anomalous graph data, are sensitive to anomaly labels, struggle to capture features of anomalous samples effectively, exhibit poor model generalization, and suffer from performance reversal issues. There is a need for improvement in anomaly detection capabilities. This paper proposes an anomaly-aware variational graph autoencoder based graph-level anomaly detection algorithm(VGAE-D), which utilizes an anomaly aware variational graph autoencoder to simultaneously extract features from normal and anomalous graph data. The model distinguishes the encoding information of normal and anomalous graphs in the encoding space, further exploring the graph encoding information to compute anomaly scores. Experimental evaluations on eight publicly available datasets from diverse domains demonstrate the effectiveness of the proposed graph-level anomaly detection method. The results indicate that VGAE-D can efficiently identify anomalous graphs in different datasets, outperforming mainstream graph-level anomaly detection methods. Additionally, the model exhibits a capability for learning from few anomalous samples, mitigating the issue of performance inversion to a large extent.

![framework](images\framework.png)

## Usage

`pip install -r requirements.txt  `

`python main.py --dataset AIDS`

To change the dataset, you can change the dataset parameter, all data can be downloaded at [TUDataset | TUD Benchmark datasets (chrsmrrs.github.io)](https://chrsmrrs.github.io/datasets/)