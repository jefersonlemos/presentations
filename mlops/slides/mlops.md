https://aws.amazon.com/what-is/machine-learning/#what-are-machine-learning-use-cases--1v16f8s
## Are machine learning models deterministic?
If a system’s output is predictable, then it is said to be deterministic. Most software applications respond predictably to the user's action, so you can say: “If the user does this, he gets that.” However, machine learning algorithms learn through observation along with experiences. Therefore, they are probabilistic in nature. The statement now changes to: “If the user does this, there is an X% chance of that happening.”

In machine learning, determinism is a strategy used while applying the learning methods described above. Any of the supervised, unsupervised, and other training methods can be made deterministic depending on the business's desired outcomes. The research question, data retrieval, structure, and storage decisions determine if a deterministic or non-deterministic strategy is adopted.

### Deterministic vs. probabilistic approach
The deterministic approach focuses on the accuracy and the amount of data collected, so efficiency is prioritized over uncertainty. On the other hand, the non-deterministic (or probabilistic) process is designed to manage the chance factor. Built-in tools are integrated into machine learning algorithms to help quantify, identify, and measure uncertainty during learning and observation.

## What are machine learning use cases?
Let’s take a look at machine learning applications in some key industries:

Manufacturing
Healthcare and life sciences
Financial services
Retail
Media and entertainment
Computer vision






https://aws.amazon.com/what-is/machine-learning/

## What's Machine learning 
https://aws.amazon.com/what-is/machine-learning/

```
Machine learning is a type of artificial intelligence that performs data analysis tasks without explicit instructions. Machine learning technology can process large quantities of historical data, identify patterns, and predict new relationships between previously unknown data. You can perform classification and prediction tasks on documents, images, numbers, and other data types. 

For example, a financial organization could train a machine learning system to classify fraudulent and genuine transactions. The system identifies patterns in known data to accurately guess or predict whether a new transaction is genuine.
```
ML is short for Machine Learning which is a branch of artificial intelligence (AI) and computer science that leverages data and algorithms to enable AI systems to learn and improve in a manner similar to humans, progressively enhancing their accuracy over time.


## What is the difference between machine learning vs. artificial intelligence?
While the terms machine learning and artificial intelligence (AI) are used interchangeably, they are not the same. Machine learning is one of many branches of AI. While machine learning is AI, not all AI activities can be called machine learning.

Artificial intelligence is the broader concept of making machines more human-like, while machine learning is a subset focused on teaching computers to learn from data without explicit programming. AI includes everything from smart assistants like Alexa, chatbots, and image generators to robotic vacuum cleaners and self-driving cars. 

In contrast, machine learning models perform more specific data analysis tasks—like classifying documents, labeling images, or predicting the maintenance schedule of factory equipment. Machine learning technology is primarily based on mathematics and statistics, while other types of AI are more complex. Essentially, machine learning is a method to achieve artificial intelligence.

## How does machine learning work?

The central idea behind machine learning is an existing mathematical relationship between any input and output data combination. The machine learning model does not know this relationship in advance but can guess if sufficient examples of input-output data sets are given. This means every machine learning algorithm is built around a modifiable math function. The underlying principle can be understood like this:

We ‘train’ the algorithm by giving it the following input/output (i,o) combinations – (2,10), (5,19), and (9,31)
The algorithm computes the relationship between input and output to be: o=3*i+4
We then give it input 7 and ask it to predict the output. It can automatically determine the output as 25.
While this is a basic understanding, machine learning focuses on the principle that computer systems can mathematically link all complex data points as long as they have sufficient data and computing power to process. Therefore, the accuracy of the output is directly co-relational to the magnitude of the input given. Machine learning phases are given below.

# 1. What's MLOps

## 1. Principles

As machine learning and AI propagate in software products and services, we need to establish best practices and tools to test, deploy, manage, and monitor ML models in real-world production. In short, with MLOps we strive to avoid _“technical debt”_ in machine learning applications.

SIG MLOps defines _“an optimal MLOps experience [as] one where Machine Learning assets are treated consistently with all other software assets within a CI/CD environment. Machine Learning models can be deployed alongside the services that wrap them and the services that consume them as part of a unified release process.”_ By codifying these practices, we hope to accelerate the adoption of ML/AI in software systems and fast delivery of intelligent software. In the following, we describe a set of important concepts in MLOps such as _Iterative-Incremental Development, Automation, Continuous Deployment, Versioning, Testing, Reproducibility, and Monitoring_.

- **Continuous Integration (CI)** extends the testing and validating code and components by adding testing and validating data and models.
- **Continuous Delivery (CD)** concerns with delivery of an ML training pipeline that automatically deploys another the ML model prediction service.
- **Continuous Training (CT)** is unique to ML systems property, which automatically retrains ML models for re-deployment.
- **Continuous Monitoring (CM)** concerns with monitoring production data and model performance metrics, which are bound to business metrics.

---
## 1.2 Components

![[Pasted image 20260516082638.png]]


---
## 1.3 Experiments Tracking

Machine Learning development is a highly iterative and research-centric process. In contrast to the traditional software development process, in ML development, multiple experiments on model training can be executed in parallel before making the decision what model will be promoted to production.

The experimentation during ML development might have the following scenario: One way to track multiple experiments is to use different (Git-) branches, each dedicated to the separate experiment. The output of each branch is a trained model. Depending on the selected metric, the trained ML models are compared with each other and the appropriate model is selected. Such low friction branching is fully supported by the tool [DVC](https://dvc.org/), which is an extension of Git and an open-source version control system for machine learning projects. Another popular tool for ML experiments tracking is the [Weights and Biases (wandb)](https://www.wandb.com/) library, which automatically tracks the hyperparameters and metrics of the experiments.

---
## 1.4 Tests
Something
Read the Testing section on [MLOps Principles](https://ml-ops.org/content/mlops-principles.html)

Read also the ML infrastructure test section

---

## 1.5 Monitoring

Read the Monitoring section on [MLOps Principles](https://ml-ops.org/content/mlops-principles.html)
# 2. Flow

![[Pasted image 20260516082132.png]]

---
The first phase is devoted to _business understanding, data understanding_ and _designing the ML-powered software_. In this stage, we identify our potential user, design the machine learning solution to solve its problem, and assess the further development of the project. Mostly, we would act within two categories of problems - either increasing the productivity of the user or increasing the interactivity of our application.

Initially, we define ML use-cases and prioritize them. The best practice for ML projects is to work on one ML use case at a time.

---

Furthermore, the _design_ phase aims to inspect the available data that will be needed to train our model and to specify the functional and non-functional requirements of our ML model. We should use these requirements to design the architecture of the ML-application, establish the serving strategy, and create a test suite for the future ML model

The follow-up phase _“ML Experimentation and Development”_ is devoted to verifying the applicability of ML for our problem by implementing _Proof-of-Concept for ML Model_. Here, we run iteratively different steps, such as _identifying or polishing the suitable ML algorithm for our problem, data engineering_, and _model engineering_. The primary goal in this phase is to deliver a stable quality ML model that we will run in production.


---

The main focus of the _“ML Operations”_ phase is to deliver the previously developed ML model in production by using established DevOps practices such as testing, versioning, continuous delivery, and monitoring.

---

![[Pasted image 20260516082510.png]]

---

1. **Manual process.** This is a typical data science process, which is performed at the beginning of implementing ML. This level has an experimental and iterative nature. Every step in each pipeline, such as data preparation and validation, model training and testing, are executed manually. The common way to process is to use Rapid Application Development (RAD) tools, such as Jupyter Notebooks.
2. **ML pipeline automation.** The next level includes the execution of model training automatically. We introduce here the continuous training of the model. Whenever new data is available, the process of model retraining is triggered. This level of automation also includes data and model validation steps.
3. **CI/CD pipeline automation.** In the final stage, we introduce a CI/CD system to perform fast and reliable ML model deployments in production. The core difference from the previous step is that we now automatically build, test, and deploy the Data, ML Model, and the ML training pipeline components.

---
# Possibilidades

## customer churn

## Treinar uma LLM
Não, LLMOps e MLOps não são coisas muito diferentes. O LLMOps é, na verdade, uma evolução direta e uma subárea do MLOps.
A principal diferença está no foco: enquanto o MLOps tradicional lida com dados estruturados e treinamento de modelos próprios, o LLMOps lida com modelos massivos pré-treinados, engenharia de prompts e saídas de texto imprevisíveis.


É possível usar uma LLM para uma apresentação de MLOps?
Sim, é totalmente possível e recomendado. Usar uma LLM em uma demonstração de MLOps é uma excelente forma de atrair a atenção do público, pois os resultados são visíveis e fáceis de entender imediatamente.
Para que o experimento faça sentido em uma apresentação de MLOps, você não deve focar no chat em si, mas sim no processo de engenharia ao redor dele.
Ideias de Experimentos para Apresentar
Aqui estão três ideias de demonstrações práticas que conectam os dois mundos:
1. Monitoramento de Custos e Latência (Observabilidade)
O experimento: Crie um script simples que faz perguntas a uma LLM (como o GPT-4o ou Claude).
O foco em MLOps: Use uma ferramenta como LangSmith ou Phoenix (Arize) para mostrar um painel em tempo real. Exiba ao público o custo de cada pergunta em centavos, o tempo de resposta (latência) e como uma pergunta longa consome mais recursos.
2. Teste de Regressão de Prompts (CI/CD)
O experimento: Mostre o que acontece quando você muda uma instrução no sistema (ex: mudar de "Seja formal" para "Seja um pirata").
O foco em MLOps: Demonstre um pipeline automatizado que testa o novo prompt contra 5 perguntas padrão antes de ir para a apresentação. Mostre o relatório comparando as duas versões para garantir que a mudança não quebrou o sistema.
3. Filtro de Segurança e Moderação (Guardrails)
O experimento: Tente fazer a LLM vazar informações fingindo ser um hacker (prompt injection).
O foco em MLOps: Mostre uma camada de código intermediária (como o NeMo Guardrails) bloqueando a tentativa antes que ela chegue ao modelo. Isso demonstra o pilar de governança e segurança de modelos do MLOps.