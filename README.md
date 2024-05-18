## 1 项目背景
随着互联网行业的发展，一些公司的业务变得非常庞大和复杂，很难通过传统的发展模式进行有效的管理。微服务架构将应用构建为围绕业务领域建模的微自治服务的集合，这些服务可以独立开发、管理和加速。近年来，微服务已经成为互联网系统架构的重要组成部分，并在许多云计算应用中得到应用。Amazon、Netflix、Twitter、PayPal、腾讯、京东、淘宝、百度等使用微服务架构对集团业务进行解构，构建微服务组件平台。微服务作为一种架构风格正被越来越多的人接受，用于在内部和内部创建大规模的基于云的应用程序。这种基于微服务的架构大大提高了系统的可扩展性，但它也引入了新的问题，包括微服务应用程序的高效调度和微服务的数据安全性。



### 1.2 面对挑战
在微服务架构中，数据保护是一个非常重要的问题。首先，由于微服务通常由多个不同的服务组成，每个服务都有自己的数据存储和处理方法。因此，需要对每个业务的数据进行适当的保护，防止数据泄露、数据丢失或非法访问。然后，微服务架构中的服务通常通过网络进行通信。这需要确保服务之间的通信是安全的，以防止数据在传输过程中被窃取或篡改。基于传统云应用的任务调度已经有很多研究。然而，与传统的工作流任务相比，面向微服务架构的工作流任务要小得多，但数量要多得多。因此，考虑到预算限制、微服务任务之间的偏序关系以及如此庞大的微服务规模，在异构云环境下优化调度完成时间具有挑战性。
![image](https://github.com/qkimmichq/Secure-Scheduling-in-cloud-environments/blob/main/IMG/%E5%9B%BE%E7%89%871.png)
### 1.3 解决方案
针对以上两个挑战，我们首先提出了敏感微服务的安全模型。根据不同的安全需求，微服务被划分为不同的安全级别。然后将它们调度到能够满足其安全需求的云资源上。其次，容器技术是一种轻量级的虚拟化技术。它的内核机制提供了容器之间的桥梁。同时，容器之间的资源是相互独立、相互隔离的。另一方面，随着人工智能的发展，机器学习已经在学术界和工业界的各个领域得到了应用。强化学习是一种很有影响力的机器学习算法，它通过与环境的交互来获得最优解。因此，我们采用容器技术和强化学习算法对微服务的工作流调度进行优化，进一步提高资源利用率。
