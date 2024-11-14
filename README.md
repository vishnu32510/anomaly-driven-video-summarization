# <a name="_uadaxqh83qq5"></a>**Project Proposal**

### <a name="_cb9h3wjs49sk"></a>**Title:** Anomaly-Driven Video Summarization for Real-Time Surveillance Systems
### <a name="_lferb7p2o0y2"></a>**Team:**
- Vishnu Priyan Sellam Shanmugavel - A20561323
- Akash Thirumuruganantham - A20539883
### <a name="_e7k9oq1qjy71"></a>**Main Paper:**
### <a name="_7xa8mhde4tfb"></a>**Name: [Real-world Anomaly Detection in Surveillance Videos**](https://arxiv.org/pdf/1801.04264v3)** [2019]
**Authors:** Waqas Sultani , Chen Chen , Mubarak Shah
### <a name="_toctlcawg1gf"></a>**Additional Paper:**
### <a name="_5p5p07tsf7uy"></a>**Name: [A Three-Stage Anomaly Detection Framework for Traffic Videos**](https://onlinelibrary.wiley.com/doi/epdf/10.1155/2022/9463559)** [2022]
**Authors:** Junzhou Chen, Jiancheng Wang, Jiajun Pu, and Ronghui Zhang
### <a name="_g2a6b88oc4px"></a>**Problem Statement:** 
`	`Surveillance systems generate large volumes of video data, which are typically collected and stored passively, leading to inefficiencies in monitoring and retrieval. The manual review of such video data is labor-intensive and prone to errors. The key challenge is to automatically detect anomalies within the videos and summarize the most relevant footage for efficient storage and quick threat identification. This project addresses how to optimize video processing to detect and summarize anomalous activities in surveillance footage without requiring real-time analysis.
### <a name="_hhqt65a7g0we"></a>**Approach:** 
To tackle the problem, we propose an anomaly detection model for surveillance videos that automatically identifies and summarizes anomalous events. The solution involves:

1. **Shot Segmentation:** Divide the video into temporal segments for detailed analysis.
1. **Feature Extraction:** Utilize both C3D and 3D ResNet to extract spatial and temporal features from the video segments, enhancing feature representation.
1. **Loss Function Modification:** Incorporate Temporal Smoothness Loss with the ranking loss function to improve anomaly localization and account for the temporal structure of the videos.
1. **Augmentation Techniques**: We will apply sparsity constraints to the loss function during training. This will enhance the model's performance by ensuring that the scores for anomalous segments are distinctly higher compared to normal segments, thereby refining the detection capabilities.
1. **Scoring Mechanism:** Develop a scoring system based on memorability, entropy, and temporal dynamics to evaluate the significance of each segment.
1. **Ranking Shots:** Sort the video segments based on their scores to identify the most relevant shots.
1. **Video Summarization:** Select keyframes from the ranked shots to create a summarized version of the video, ensuring efficient retrieval and monitoring.
1. **Datasets:** The project will utilize two datasets: **UFC-Crime** for real-world anomaly detection in crime-related incidents and **another specified dataset** for supplementary analysis.
### <a name="_haxj99nfmlck"></a>**Data:**
- [UFC-Crime](https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset): The dataset contains extracted images from the UCF crime dataset used for Real-world Anomaly Detection in Surveillance Videos
- [UBNormal](https://drive.google.com/file/d/1KbfdyasribAMbbKoBU1iywAhtoAt9QI0/view): A widely used surveillance video dataset for anomaly detection
- [UBI-Flights](https://socia-lab.di.ubi.pt/EventDetection/):  A comprehensive collection of real-world crime videos.
### <a name="_p85aubsnaxch"></a>**Which option do you choose?**
We choose **Option 2:** Modification done to the loss function incorporate Temporal Smoothness Loss, use both C3D and 3D ResNet to extract feature and addition of Video Summarization with Key frames

### <a name="_41u7fdb2bf55"></a>**Team Member Responsibilities:**
- **Vishnu Priyan:**
  - Feature extraction and implementation of C3D and 3D ResNet.
  - Loss function modification and scoring mechanism development.
- **Akash:**
  - Video segmentation and summarization process.
  - Dataset preparation and evaluation.
