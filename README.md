# Agro-Rover

This project introduces an AI-powered smart rover designed for the early detection of tomato leaf diseases, addressing the inefficiencies of manual inspection in large-scale agriculture. By identifying threats like Early Blight and Leaf Mould in their infancy, the system helps farmers protect crop yields and minimize economic loss.

# System Architecture & Technology
The solution integrates a mobile rover platform with high-performance hardware and software:
**Core Hardware:** A Raspberry Pi serves as the central processing unit, paired with a high-definition webcam for visual data acquisition.
**AI Framework:** A machine learning model developed using the **Ultralytics YOLO** (PyTorch) framework provides rapid and accurate image classification.
**Interface:** The system utilizes the **Expo Go** app, providing a seamless mobile interface for communicating with the rover and monitoring data.

# Operational Workflow
As the rover navigates the field, it streams a live video feed of tomato foliage. The integrated AI model processes this data on demand to distinguish between healthy and infected leaves. When a disease is identified, the **Expo Go** interface updates instantly, displaying the **specific disease type** and the **AI’s confidence level**. This real-time feedback allows for immediate remote intervention.

#Impact and Sustainability
By merging embedded systems, computer vision, and IoT, this rover offers a scalable and cost-effective tool for **precision agriculture**. The ability to perform continuous, real-time diagnostics enables farmers to take immediate corrective action, preventing widespread infection and promoting more sustainable, productive farming practices.
