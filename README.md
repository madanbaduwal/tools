# tools

## Python 

- [X] [PyChecker]()
- [X] Profiling Tools
	- [X] [cProfile](https://docs.python.org/3.2/library/profile.html)
	- [X] [Profile](https://docs.python.org/3.2/library/profile.html)
	- [X] [Pympler](https://pythonhosted.org/Pympler/)
	- [X] [Objgraph](https://mg.pov.lt/objgraph/)
	- [X] [pyinstrument](https://github.com/joerick/pyinstrument)
- [X] [Regex For Noobs (like me!) - An Illustrated Guide](https://www.janmeppe.com/blog/regex-for-noobs/)
- [X] [argparse](https://docs.python.org/3/library/argparse.html): Write user-friendly command-line interfaces  
- [X] [beautifulsoup](https://pypi.org/project/beautifulsoup4/): Pull data out of HTML and XML files  
- [X] [black](https://github.com/psf/black): Opiniated code formatter for python code  
- [boto/boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html): Control AWS service with pure python code  
- [X] [conda](https://docs.conda.io/en/latest/): Package, dependency and environment management  
- [X] [datetime](https://docs.python.org/3/library/datetime.html): Supplies classes for manipulating dates and times 
- [X] [fastai](https://pypi.org/project/fastai/): fastai makes deep learning with PyTorch faster, more accurate, and easier  
- [X] [gspread](https://github.com/burnash/gspread): Python library to interact with Google Sheets  
- [X] [gunicorn](https://pypi.org/project/gunicorn/): Production web server for Flask, Django apps  
- [X] [ipython](https://pypi.org/project/ipython/): IPython: Productive Interactive Computing  
- [X] [itertools](https://docs.python.org/2/library/itertools.html): Functions creating iterators for efficient looping  
- [X] [json](https://docs.python.org/3/library/json.html): Read and write JSON files  
- [X] [jupyter](https://pypi.org/project/jupyter/): Jupyter notebooks  
- [X] [jupyterlab](https://pypi.org/project/jupyterlab/): An extensible environment for interactive and reproducible computing, based on the Jupyter Notebook and Architecture  
- [X] [memory-profiler](https://pypi.org/project/memory-profiler/): A module for monitoring memory usage of a python program  
- [X] [mongoengine](https://pypi.org/project/mongoengine/): MongoEngine is a Python Object-Document Mapper for working with MongoDB.  
- [X] [more_itertools](https://pypi.org/project/more-itertools/): More routines for operating on iterables, beyond itertools  
- [X] [multiprocessing-logging](https://pypi.org/project/multiprocessing-logging/): Logger for multiprocessing applications  
- [X] [xlrd](https://pypi.org/project/xlrd/): Extract data from Excel spreadsheets  
- [X] [yaml](https://pyyaml.org/wiki/PyYAMLDocumentation): YAML parser and emitter for Python  	
- [X] [pipenv](https://pypi.org/project/pipenv/): Pipenv is a tool that aims to bring the best of all packaging worlds (bundler, composer, npm, cargo, yarn, etc.) to the Python world.  
- [X] [pymongo](https://pypi.org/project/pymongo/): Python driver for MongoDB  
- [X] [pymysql](https://pypi.org/project/PyMySQL/): Pure Python MySQL Driver  
- [X] [pypdf2](https://pypi.org/project/PyPDF2/): PDF toolkit  
- [X] [pyspark](https://pypi.org/project/pyspark/): Apache Spark Python API  
- [X] [pytest](https://pypi.org/project/pytest/): pytest: simple powerful testing with Python  
- [X] [python-dotenv](https://pypi.org/project/python-dotenv/): Add .env support to your django/flask apps in development and deployments  
- [X] [pyyaml](https://pypi.org/project/PyYAML/): YAML 1.1 parser  
- [X] [rasterio](https://pypi.org/project/rasterio/): Reads and writes GeoTIFF formats and provides a Python API based on N-D arrays  
- [X] [re](https://docs.python.org/3/library/re.html): Regular expression matching operations  
- [X] [requests](https://pypi.org/project/requests/): HTTP library for Python   
- [X] [sqlalchemy](https://www.sqlalchemy.org/): Python SQL toolkit  
- [X] [tabulapy](https://pypi.org/project/tabula-py/): Python wrapper of tabula-java, which can read table of PDF   
- [X] [urllib](https://docs.python.org/3/library/urllib.html): Collects several modules for working with URLs  
- [X] [awesome-python:Python Library](https://github.com/vinta/awesome-python)



## Databases

- [Schema design]()

**Key-Value Database**: Picture a colossal locker room. Each locker (value) has a unique key. Storing and retrieving data becomes a cakewalk! Examples: Amazon QLDB, Redis, AWS DynamoDB.
- [X] [Redis](https://redis.io/docs/)
- [X] [AWS DynamoDB](https://docs.aws.amazon.com/dynamodb/index.html)
- [X] [apache Hbase](https://hbase.apache.org/book.html)

 **Document Database**: Store data in a flexible, document-like format (similar to JSON). MongoDB and Couchbase excel in this space, offering efficient querying and flexibility.

- [X] [MongoDB](https://www.mongodb.com/docs/)
    - [X] [mongo compass]()
	- [X] [studio 3T]()
	- [X] [Pymongo]()

- [X] [Couchbase/CouchDB](https://docs.couchbase.com/home/index.html)


**Relational database/SQL(RDBMS)**: The classical choice! SQL databases store structured data in tables and support powerful queries. MySQL, Oracle, and Microsoft SQL Server are stalwarts in this space. Relational databases are also called a relational database management system (RDBMS) or SQL database.
  - [X] [MYSQL](https://dev.mysql.com/doc/)
  - [X] [Oracle](https://www.oracle.com/database/)
  - [X] [Microsoft SQL Server](https://learn.microsoft.com/en-us/sql/sql-server/?view=sql-server-ver16)
  - [X] [PostgreSQL](https://www.postgresql.org/docs/)
  - [X] [SQLite](https://www.sqlite.org/index.html)

**Columnar Databases**: These databases store data by columns, not rows. Ideal for analytical queries and efficient data compression. Apache Cassandra and DataStax are well-known examples.


- [X] [Cassendra](https://cassandra.apache.org/_/index.html)
- [X] [DataStax](https://docs.datastax.com/en/home/docs/index.html)

**NewSQL**: It's the perfect blend of NoSQL scalability with SQL's reliability. Key players include CockroachDB, VoltDB, and NuoDB.

- [X] [CockroachDB](https://www.cockroachlabs.com/docs/stable/)
- [X] [VoltDB](https://docs.voltdb.com/)
- [X] [NuoDB](https://doc.nuodb.com/nuodb/latest/release-notes/)
- [X] [SinglesStore](https://docs.singlestore.com/)
- [X] [clustrix](https://dbdb.io/db/clustrix)


**Graph Database**: The social network of databases! They're champions in delivering deep insights about connections. Think LinkedIn network analysis with Neo4j or AWS Neptune.
- [X] [Neo4j](https://neo4j.com/)
- [X] [AWS Neptune](https://docs.aws.amazon.com/neptune/index.html)
- [X] [Janus Graph](https://docs.janusgraph.org/)

**Time-Series Databases**: A dream for tracking changes over time, like stock prices or weather data. InfluxDB and Prometheus are top picks for their superior query performance.

- [X] [InfluxDB](https://docs.influxdata.com/)
- [X] [KairosDB](https://kairosdb.github.io/docs/index.html)
- [X] [Prometheus](https://prometheus.io/docs/introduction/overview/)
- [X] [ClickHouse](https://clickhouse.com/docs/en/intro)

**Spatial Databases**: Geography and technology entwine! These databases store geographic data (like coordinates for landmarks or cities). They're the powerhouse behind Google Maps and Uber!

- [X] [Snowflake](https://docs.snowflake.com/)
- [X] [Oracle](https://docs.oracle.com/en/)
- [X] [Microsoft SQL Server ](https://learn.microsoft.com/en-us/sql/sql-server/?view=sql-server-ver16)

**Ledger Database**: These databases stand for transparency and immutability. They're the backbone of blockchain technologies, with Apache HBase being a notable example.
- [X] [Amazon QLDB](https://docs.aws.amazon.com/qldb/index.html)


**Object Databases**: Data as objects, mirroring object-oriented programming. ObjectDB and ZODB provide encapsulation, inheritance, polymorphism, and more.

- [X] [ObjectDB](https://www.objectdb.com/database/jdo/manual)
- [X] [ZODB](https://zodb.org/en/latest/)

**Vector Database**
- [X] [vdbs](https://vdbs.superlinked.com/)

**Data Lakes**

- [X] [Snowflake](https://docs.snowflake.com/)

- [X] [Databricks](https://docs.databricks.com/index.html)

- [X] [Cloudera](https://docs.cloudera.com/management-console/cloud/data-lakes/topics/mc-data-lake.html)

**Data Warehouses**
- [X] [Amazon Redshift](https://docs.aws.amazon.com/redshift/index.html)

- [X] [Google Cloud BigQuery](https://cloud.google.com/bigquery/docs)

- [X] [Snowflake](https://docs.snowflake.com/)

- [X] [Microsoft Azure](https://learn.microsoft.com/en-us/azure/architecture/data-guide/relational-data/data-warehousing)

- [X] [IBM Db2](https://www.ibm.com/docs/en/db2-warehouse)

- [X] [Teradata](https://www.teradata.com/Cloud/Data-Warehouse)


**Data Warehouses vs Data Lake**

- Data lake: A centralized repository that stores large amounts of structured, semi-structured, and unstructured data
- Data warehouse: A data management system that stores structured, processed, and refined data
- Data type: Data lakes store raw, unprocessed data, such as multimedia files, log files, and other large files
- Purpose: Data warehouses store data that has been treated and transformed for a specific purpose, such as analytic or operational reporting
- Use: Data warehouses are designed to enable business intelligence activities, such as analytics
- Overlap: Data lakes and data warehouses are supplemental technologies that serve different use cases, but there is some overlap


**Files/Storage**


**Miscellinious**

- [X] [ElasticSearch](https://www.elastic.co/)


#### Management

- [X] [ Airflow](https://airflow.apache.org/docs/apache-airflow/stable/index.html)


## Machine learning and deep learning

* [Huggingface](https://huggingface.co/) | [Doc](https://huggingface.co/docs)
    * Upload model
    * Upload data
    * Space for interactive app
    * Deploy your model in sagemaker
    * Write webhooks for MLOPS
    * Pull request and discussion : 
        * File and version > new pull request
        Note: Under the hood, our Pull requests do not use forks and branches, but instead, custom "branches" called refs that are stored directly on the source repo.
    * Automatic fine tune with auto train
* [Pytorch] | [Documentation]()
- [X] [Numpy](https://numpy.org/doc/)
- [X] [Sklearn](https://scikit-learn.org/stable/)
- [X] [Pytorch](https://pytorch.org/docs/stable/index.html): The torch package contains data structures for multi-dimensional tensors and defines mathematical operations over these tensors. Additionally, it provides many utilities for efficient serializing of Tensors and arbitrary types, and other useful utilities.
    - layers (which in modern machine learning should really be understood as stateful functions with implicit parameters) are typically expressed as Python classes whose constructors create and initialize their parameters, and whose forward methods process an input activation. 
    - **To define a neural network in PyTorch, we create a class that inherits from nn.Module. We define the layers of the network in the __init__ function and specify how data will pass through the network in the forward function.**
    - **Tensors** : are similar to NumPy’s ndarrays, except that tensors can run on GPUs or other hardware accelerators.
    - **model.train()**: In PyTorch, the model.train() method is commonly used to set a neural network model into **training mode(model.train() don't start training it just prepared for training)**. When you call model.train(), it puts the model in a state where it's prepared to update its weights based on the gradients computed during the training process. This is important because it activates certain training-specific features in the model, such as dropout and batch normalization layers, which behave differently during training compared to evaluation or inference.
    - **model.eval()**: 
    - Grenerl code for neural network.
  	```python
	
    # Import libraries
  	import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from torchvision.transforms import ToTensor

	# Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

	batch_size = 64

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

	# Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # Define model
    class NeuralNetwork(nn.Module):
        def __init__(self): # Define layers
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10)
            )

        def forward(self, x): # Pass data to the layers
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    model = NeuralNetwork().to(device)
    print(model)
    
	# To train a model, we need a loss function and an optimizer.
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

	# Train a model
	def train(dataloader, model, loss_fn, optimizer):
      size = len(dataloader.dataset)
      model.train() # just prepared for training
      for batch, (X, y) in enumerate(dataloader):
          X, y = X.to(device), y.to(device)

          # Compute prediction error
          pred = model(X)
          loss = loss_fn(pred, y)

          # Backpropagation
          loss.backward()
          optimizer.step()
          optimizer.zero_grad()

          if batch % 100 == 0:
              loss, current = loss.item(), (batch + 1) * len(X)
              print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

	# Test a model
	def test(dataloader, model, loss_fn):
      size = len(dataloader.dataset)
      num_batches = len(dataloader)
      model.eval()  # just prepared for evaluation
      test_loss, correct = 0, 0
      with torch.no_grad():
          for X, y in dataloader:
              X, y = X.to(device), y.to(device)
              pred = model(X)
              test_loss += loss_fn(pred, y).item()
              correct += (pred.argmax(1) == y).type(torch.float).sum().item()
      test_loss /= num_batches
      correct /= size
      print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
	
    
	# The training process is conducted over several iterations (epochs).
	epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")

	# A common way to save a model is to serialize the internal state dictionary (containing the model parameters).
	torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")

	# The process for loading a model includes re-creating the model structure and loading the state dictionary into it.
	model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load("model.pth"))

	# This model can now be used to make predictions.
	classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    model.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')
  	
  	``` 
	- In PyTorch, we use tensors to encode the inputs and outputs of a model, as well as the model’s parameters.
	- 
- [ ] [REXMEX](https://rexmex.readthedocs.io/en/latest/)
- [X] [bert-as-a-service](https://github.com/hanxiao/bert-as-service): Generate BERT Embeddings for production  
- [X] [camelot](https://github.com/socialcopsdev/camelot): Extract tables from PDF files  
- [X] [deepctr](https://pypi.org/project/deepctr/): Deep-learning based CTR models  
- [X] [dlib](https://pypi.org/project/dlib/): A toolkit for making real world machine learning and data analysis applications in C++  
- [X] [docx2txt](https://pypi.org/project/docx2txt/): A pure python-based utility to extract text and images from docx files
 * [X] [RexMex](https://github.com/AstraZeneca/rexmex) -> A general purpose recommender metrics library for fair evaluation.
 * [X] [ChemicalX](https://github.com/AstraZeneca/chemicalx) -> A PyTorch based deep learning library for drug pair scoring
 * [X] [Microsoft ML for Apache Spark](https://github.com/Azure/mmlspark) -> A distributed machine learning framework Apache Spark
 * [X] [Shapley](https://github.com/benedekrozemberczki/shapley) -> A data-driven framework to quantify the value of classifiers in a machine learning ensemble.
 * [X] [igel](https://github.com/nidhaloff/igel) -> A delightful machine learning tool that allows you to train/fit, test and use models **without writing code**
 * [X] [ML Model building](https://github.com/Shanky-21/Machine_learning) -> A Repository Containing Classification, Clustering, Regression, Recommender Notebooks with illustration to makeNew-Grad-Positions them.
 * [X] [ML/DL project template](https://github.com/PyTorchLightning/deep-learning-project-template)
 * [X] [PyTorch Geometric Temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal) -> A temporal extension of PyTorch Geometric for dynamic graph representation learning.
 * [X] [Little Ball of Fur](https://github.com/benedekrozemberczki/littleballoffur) -> A graph sampling extension library for NetworkX with a Scikit-Learn like API.
 * [X] [Karate Club](https://github.com/benedekrozemberczki/karateclub) -> An unsupervised machine learning extension library for NetworkX with a Scikit-Learn like API.
* [X] [Auto_ViML](https://github.com/AutoViML/Auto_ViML) -> Automatically Build Variant Interpretable ML models fast! Auto_ViML is pronounced "auto vimal", is a comprehensive and scalable Python AutoML toolkit with imbalanced handling, ensembling, stacking and built-in feature selection. Featured in <a href="https://towardsdatascience.com/why-automl-is-an-essential-new-tool-for-data-scientists-2d9ab4e25e46?source=friends_link&sk=d03a0cc55c23deb497d546d6b9be0653">Medium article</a>.
* [X] [PyOD](https://github.com/yzhao062/pyod) -> Python Outlier Detection, comprehensive and scalable Python toolkit for detecting outlying objects in multivariate data. Featured for Advanced models, including Neural Networks/Deep Learning and Outlier Ensembles.
* [X] [steppy](https://github.com/neptune-ml/steppy) -> Lightweight, Python library for fast and reproducible machine learning experimentation. Introduces a very simple interface that enables clean machine learning pipeline design.
* [X] [steppy-toolkit](https://github.com/neptune-ml/steppy-toolkit) -> Curated collection of the neural networks, transformers and models that make your machine learning work faster and more effective.
* [X] [CNTK](https://github.com/Microsoft/CNTK) - Microsoft Cognitive Toolkit (CNTK), an open source deep-learning toolkit. Documentation can be found [here](https://docs.microsoft.com/cognitive-toolkit/).
* [X] [Couler](https://github.com/couler-proj/couler) - Unified interface for constructing and managing machine learning workflows on different workflow engines, such as Argo Workflows, Tekton Pipelines, and Apache Airflow.
* [X] [auto_ml](https://github.com/ClimbsRocks/auto_ml) - Automated machine learning for production and analytics. Lets you focus on the fun parts of ML, while outputting production-ready code, and detailed analytics of your dataset and results. Includes support for NLP, XGBoost, CatBoost, LightGBM, and soon, deep learning.
* [X] [dtaidistance](https://github.com/wannesm/dtaidistance) - High performance library for time series distances (DTW) and time series clustering.
* [X] [machine learning](https://github.com/jeff1evesque/machine-learning) - automated build consisting of a [web-interface](https://github.com/jeff1evesque/machine-learning#web-interface), and set of [programmatic-interface](https://github.com/jeff1evesque/machine-learning#programmatic-interface) API, for support vector machines. Corresponding dataset(s) are stored into a SQL database, then generated model(s) used for prediction(s), are stored into a NoSQL datastore.
* [X] [XGBoost](https://github.com/dmlc/xgboost) - Python bindings for eXtreme Gradient Boosting (Tree) Library.
* [X] [ChefBoost](https://github.com/serengil/chefboost) - a lightweight decision tree framework for Python with categorical feature support covering regular decision tree algorithms such as ID3, C4.5, CART, CHAID and regression tree; also some advanved bagging and boosting techniques such as gradient boosting, random forest and adaboost.
* [X] [Apache SINGA](https://singa.apache.org) - An Apache Incubating project for developing an open source machine learning library.
* [X] [Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers) - Book/iPython notebooks on Probabilistic Programming in Python.
* [X] [Featureforge](https://github.com/machinalis/featureforge) A set of tools for creating and testing machine learning features, with a scikit-learn compatible API.
* [X] [MLlib in Apache Spark](http://spark.apache.org/docs/latest/mllib-guide.html) - Distributed machine learning library in Spark
* [X] [Hydrosphere Mist](https://github.com/Hydrospheredata/mist) - a service for deployment Apache Spark MLLib machine learning models as realtime, batch or reactive web services.
* [X] [Towhee](https://towhee.io) - A Python module that encode unstructured data into embeddings.
* [X] [scikit-learn](https://scikit-learn.org/) - A Python module for machine learning built on top of SciPy.
* [X] [metric-learn](https://github.com/metric-learn/metric-learn) - A Python module for metric learning.
* [X] [Intel(R) Extension for Scikit-learn](https://github.com/intel/scikit-learn-intelex) - A seamless way to speed up your Scikit-learn applications with no accuracy loss and code changes.
* [X] [SimpleAI](https://github.com/simpleai-team/simpleai) Python implementation of many of the artificial intelligence algorithms described in the book "Artificial Intelligence, a Modern Approach". It focuses on providing an easy to use, well documented and tested library.
* [X] [astroML](https://www.astroml.org/) - Machine Learning and Data Mining for Astronomy.
* [X] [graphlab-create](https://turi.com/products/create/docs/) - A library with various machine learning models (regression, clustering, recommender systems, graph analytics, etc.) implemented on top of a disk-backed DataFrame.
* [X] [BigML](https://bigml.com) - A library that contacts external servers.
* [X] [pattern](https://github.com/clips/pattern) - Web mining module for Python.
* [X] [NuPIC](https://github.com/numenta/nupic) - Numenta Platform for Intelligent Computing.
* [X] [Pylearn2](https://github.com/lisa-lab/pylearn2) - A Machine Learning library based on [Theano](https://github.com/Theano/Theano). **[Deprecated]**
* [X] [keras](https://github.com/keras-team/keras) - High-level neural networks frontend for [TensorFlow](https://github.com/tensorflow/tensorflow), [CNTK](https://github.com/Microsoft/CNTK) and [Theano](https://github.com/Theano/Theano).
* [X] [Lasagne](https://github.com/Lasagne/Lasagne) - Lightweight library to build and train neural networks in Theano.
* [X] [hebel](https://github.com/hannes-brt/hebel) - GPU-Accelerated Deep Learning Library in Python. **[Deprecated]**
* [X] [Chainer](https://github.com/chainer/chainer) - Flexible neural network framework.
* [X] [prophet](https://facebook.github.io/prophet/) - Fast and automated time series forecasting framework by Facebook.
* [X] [gensim](https://github.com/RaRe-Technologies/gensim) - Topic Modelling for Humans.
* [X] [topik](https://github.com/ContinuumIO/topik) - Topic modelling toolkit. **[Deprecated]**
* [X] [PyBrain](https://github.com/pybrain/pybrain) - Another Python Machine Learning Library.
* [X] [Brainstorm](https://github.com/IDSIA/brainstorm) - Fast, flexible and fun neural networks. This is the successor of PyBrain.
* [X] [Surprise](https://surpriselib.com) - A scikit for building and analyzing recommender systems.
* [X] [implicit](https://implicit.readthedocs.io/en/latest/quickstart.html) - Fast Python Collaborative Filtering for Implicit Datasets.
* [X] [LightFM](https://making.lyst.com/lightfm/docs/home.html) -  A Python implementation of a number of popular recommendation algorithms for both implicit and explicit feedback.
* [X] [Crab](https://github.com/muricoca/crab) - A flexible, fast recommender engine. **[Deprecated]**
* [X] [python-recsys](https://github.com/ocelma/python-recsys) - A Python library for implementing a Recommender System.
* [X] [thinking bayes](https://github.com/AllenDowney/ThinkBayes) - Book on Bayesian Analysis.
* [X] [Image-to-Image Translation with Conditional Adversarial Networks](https://github.com/williamFalcon/pix2pix-keras) - Implementation of image to image (pix2pix) translation from the paper by [isola et al](https://arxiv.org/pdf/1611.07004.pdf).[DEEP LEARNING]
* [X] [Restricted Boltzmann Machines](https://github.com/echen/restricted-boltzmann-machines) -Restricted Boltzmann Machines in Python. [DEEP LEARNING]
* [X] [Bolt](https://github.com/pprett/bolt) - Bolt Online Learning Toolbox. **[Deprecated]**
* [X] [CoverTree](https://github.com/patvarilly/CoverTree) - Python implementation of cover trees, near-drop-in replacement for scipy.spatial.kdtree **[Deprecated]**
* [X] [nilearn](https://github.com/nilearn/nilearn) - Machine learning for NeuroImaging in Python.
* [X] [neuropredict](https://github.com/raamana/neuropredict) - Aimed at novice machine learners and non-expert programmers, this package offers easy (no coding needed) and comprehensive machine learning (evaluation and full report of predictive performance WITHOUT requiring you to code) in Python for NeuroImaging and any other type of features. This is aimed at absorbing much of the ML workflow, unlike other packages like nilearn and pymvpa, which require you to learn their API and code to produce anything useful.
* [X] [imbalanced-learn](https://imbalanced-learn.org/stable/) - Python module to perform under sampling and oversampling with various techniques.
* [X] [imbalanced-ensemble](https://github.com/ZhiningLiu1998/imbalanced-ensemble) - Python toolbox for quick implementation, modification, evaluation, and visualization of ensemble learning algorithms for class-imbalanced data. Supports out-of-the-box multi-class imbalanced (long-tailed) classification.
* [X] [Shogun](https://github.com/shogun-toolbox/shogun) - The Shogun Machine Learning Toolbox.
* [X] [Pyevolve](https://github.com/perone/Pyevolve) - Genetic algorithm framework. **[Deprecated]**
* [X] [Caffe](https://github.com/BVLC/caffe) - A deep learning framework developed with cleanliness, readability, and speed in mind.
* [X] [breze](https://github.com/breze-no-salt/breze) - Theano based library for deep and recurrent neural networks.
* [X] [Cortex](https://github.com/cortexlabs/cortex) - Open source platform for deploying machine learning models in production.
* [X] [pyhsmm](https://github.com/mattjj/pyhsmm) - library for approximate unsupervised inference in Bayesian Hidden Markov Models (HMMs) and explicit-duration Hidden semi-Markov Models (HSMMs), focusing on the Bayesian Nonparametric extensions, the HDP-HMM and HDP-HSMM, mostly with weak-limit approximations.
* [X] [SKLL](https://github.com/EducationalTestingService/skll) - A wrapper around scikit-learn that makes it simpler to conduct experiments.
* [X] [neurolab](https://github.com/zueve/neurolab)
* [X] [Spearmint](https://github.com/HIPS/Spearmint) - Spearmint is a package to perform Bayesian optimization according to the algorithms outlined in the paper: Practical Bayesian Optimization of Machine Learning Algorithms. Jasper Snoek, Hugo Larochelle and Ryan P. Adams. Advances in Neural Information Processing Systems, 2012. **[Deprecated]**
* [X] [Pebl](https://github.com/abhik/pebl/) - Python Environment for Bayesian Learning. **[Deprecated]**
* [X] [Theano](https://github.com/Theano/Theano/) - Optimizing GPU-meta-programming code generating array oriented optimizing math compiler in Python.
* [X] [TensorFlow](https://github.com/tensorflow/tensorflow/) - Open source software library for numerical computation using data flow graphs.
* [X] [pomegranate](https://github.com/jmschrei/pomegranate) - Hidden Markov Models for Python, implemented in Cython for speed and efficiency.
* [X] [python-timbl](https://github.com/proycon/python-timbl) - A Python extension module wrapping the full TiMBL C++ programming interface. Timbl is an elaborate k-Nearest Neighbours machine learning toolkit.
* [X] [deap](https://github.com/deap/deap) - Evolutionary algorithm framework.
* [pydeep](https://github.com/andersbll/deeppy) - Deep Learning In Python. **[Deprecated]**
* [X] [mlxtend](https://github.com/rasbt/mlxtend) - A library consisting of useful tools for data science and machine learning tasks.
* [X] [neon](https://github.com/NervanaSystems/neon) - Nervana's [high-performance](https://github.com/soumith/convnet-benchmarks) Python-based Deep Learning framework [DEEP LEARNING]. **[Deprecated]**
* [X] [Optunity](https://optunity.readthedocs.io/en/latest/) - A library dedicated to automated hyperparameter optimization with a simple, lightweight API to facilitate drop-in replacement of grid search.
* [X] [Neural Networks and Deep Learning](https://github.com/mnielsen/neural-networks-and-deep-learning) - Code samples for my book "Neural Networks and Deep Learning" [DEEP LEARNING].
* [X] [Annoy](https://github.com/spotify/annoy) - Approximate nearest neighbours implementation.
* [X] [TPOT](https://github.com/EpistasisLab/tpot) - Tool that automatically creates and optimizes machine learning pipelines using genetic programming. Consider it your personal data science assistant, automating a tedious part of machine learning.
* [X] [pgmpy](https://github.com/pgmpy/pgmpy) A python library for working with Probabilistic Graphical Models.
* [X] [DIGITS](https://github.com/NVIDIA/DIGITS) - The Deep Learning GPU Training System (DIGITS) is a web application for training deep learning models.
* [X] [Orange](https://orange.biolab.si/) - Open source data visualization and data analysis for novices and experts.
* [X] [MXNet](https://github.com/apache/incubator-mxnet) - Lightweight, Portable, Flexible Distributed/Mobile Deep Learning with Dynamic, Mutation-aware Dataflow Dep Scheduler; for Python, R, Julia, Go, Javascript and more.
* [X] [milk](https://github.com/luispedro/milk) - Machine learning toolkit focused on supervised classification. **[Deprecated]**
* [X] [TFLearn](https://github.com/tflearn/tflearn) - Deep learning library featuring a higher-level API for TensorFlow.
* [X] [REP](https://github.com/yandex/rep) - an IPython-based environment for conducting data-driven research in a consistent and reproducible way. REP is not trying to substitute scikit-learn, but extends it and provides better user experience. **[Deprecated]**
* [X] [rgf_python](https://github.com/RGF-team/rgf) - Python bindings for Regularized Greedy Forest (Tree) Library.
* [X] [skbayes](https://github.com/AmazaspShumik/sklearn-bayes) - Python package for Bayesian Machine Learning with scikit-learn API.
* [X] [fuku-ml](https://github.com/fukuball/fuku-ml) - Simple machine learning library, including Perceptron, Regression, Support Vector Machine, Decision Tree and more, it's easy to use and easy to learn for beginners.
* [X] [Xcessiv](https://github.com/reiinakano/xcessiv) - A web-based application for quick, scalable, and automated hyperparameter tuning and stacked ensembling.
* [X] [PyTorch](https://github.com/pytorch/pytorch) - Tensors and Dynamic neural networks in Python with strong GPU acceleration
* [X] [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) - The lightweight PyTorch wrapper for high-performance AI research.
* [X] [PyTorch Lightning Bolts](https://github.com/PyTorchLightning/pytorch-lightning-bolts) - Toolbox of models, callbacks, and datasets for AI/ML researchers.
* [X] [skorch](https://github.com/skorch-dev/skorch) - A scikit-learn compatible neural network library that wraps PyTorch.
* [X] [ML-From-Scratch](https://github.com/eriklindernoren/ML-From-Scratch) - Implementations of Machine Learning models from scratch in Python with a focus on transparency. Aims to showcase the nuts and bolts of ML in an accessible way.
* [X] [Edward](http://edwardlib.org/) - A library for probabilistic modeling, inference, and criticism. Built on top of TensorFlow.
* [X] [xRBM](https://github.com/omimo/xRBM) - A library for Restricted Boltzmann Machine (RBM) and its conditional variants in Tensorflow.
* [X] [CatBoost](https://github.com/catboost/catboost) - General purpose gradient boosting on decision trees library with categorical features support out of the box. It is easy to install, well documented and supports CPU and GPU (even multi-GPU) computation.
* [X] [stacked_generalization](https://github.com/fukatani/stacked_generalization) - Implementation of machine learning stacking technique as a handy library in Python.
* [X] [modAL](https://github.com/modAL-python/modAL) - A modular active learning framework for Python, built on top of scikit-learn.
* [X] [Cogitare](https://github.com/cogitare-ai/cogitare): A Modern, Fast, and Modular Deep Learning and Machine Learning framework for Python.
* [X] [Parris](https://github.com/jgreenemi/Parris) - Parris, the automated infrastructure setup tool for machine learning algorithms.
* [X] [neonrvm](https://github.com/siavashserver/neonrvm) - neonrvm is an open source machine learning library based on RVM technique. It's written in C programming language and comes with Python programming language bindings.
* [X] [Turi Create](https://github.com/apple/turicreate) - Machine learning from Apple. Turi Create simplifies the development of custom machine learning models. You don't have to be a machine learning expert to add recommendations, object detection, image classification, image similarity or activity classification to your app.
* [X] [xLearn](https://github.com/aksnzhy/xlearn) - A high performance, easy-to-use, and scalable machine learning package, which can be used to solve large-scale machine learning problems. xLearn is especially useful for solving machine learning problems on large-scale sparse data, which is very common in Internet services such as online advertisement and recommender systems.
* [X] [mlens](https://github.com/flennerhag/mlens) - A high performance, memory efficient, maximally parallelized ensemble learning, integrated with scikit-learn.
* [X] [Thampi](https://github.com/scoremedia/thampi) - Machine Learning Prediction System on AWS Lambda
* [X] [MindsDB](https://github.com/mindsdb/mindsdb) - Open Source framework to streamline use of neural networks.
* [X] [Microsoft Recommenders](https://github.com/Microsoft/Recommenders): Examples and best practices for building recommendation systems, provided as Jupyter notebooks. The repo contains some of the latest state of the art algorithms from Microsoft Research as well as from other companies and institutions.
* [X] [StellarGraph](https://github.com/stellargraph/stellargraph): Machine Learning on Graphs, a Python library for machine learning on graph-structured (network-structured) data.
* [X] [BentoML](https://github.com/bentoml/bentoml): Toolkit for package and deploy machine learning models for serving in production
* [X] [MiraiML](https://github.com/arthurpaulino/miraiml): An asynchronous engine for continuous & autonomous machine learning, built for real-time usage.
* [X] [numpy-ML](https://github.com/ddbourgin/numpy-ml): Reference implementations of ML models written in numpy
* [X] [Neuraxle](https://github.com/Neuraxio/Neuraxle): A framework providing the right abstractions to ease research, development, and deployment of your ML pipelines.
* [X] [Cornac](https://github.com/PreferredAI/cornac) - A comparative framework for multimodal recommender systems with a focus on models leveraging auxiliary data.
* [X] [JAX](https://github.com/google/jax) - JAX is Autograd and XLA, brought together for high-performance machine learning research.
* [X] [Catalyst](https://github.com/catalyst-team/catalyst) - High-level utils for PyTorch DL & RL research. It was developed with a focus on reproducibility, fast experimentation and code/ideas reusing. Being able to research/develop something new, rather than write another regular train loop.
* [X] [Fastai](https://github.com/fastai/fastai) - High-level wrapper built on the top of Pytorch which supports vision, text, tabular data and collaborative filtering.
* [X] [scikit-multiflow](https://github.com/scikit-multiflow/scikit-multiflow) - A machine learning framework for multi-output/multi-label and stream data.
* [X] [Lightwood](https://github.com/mindsdb/lightwood) - A Pytorch based framework that breaks down machine learning problems into smaller blocks that can be glued together seamlessly with objective to build predictive models with one line of code.
* [X] [bayeso](https://github.com/jungtaekkim/bayeso) - A simple, but essential Bayesian optimization package, written in Python.
* [X] [mljar-supervised](https://github.com/mljar/mljar-supervised) - An Automated Machine Learning (AutoML) python package for tabular data. It can handle: Binary Classification, MultiClass Classification and Regression. It provides explanations and markdown reports.
* [X] [evostra](https://github.com/alirezamika/evostra) - A fast Evolution Strategy implementation in Python.
* [X] [Determined](https://github.com/determined-ai/determined) - Scalable deep learning training platform, including integrated support for distributed training, hyperparameter tuning, experiment tracking, and model management.
* [X] [PySyft](https://github.com/OpenMined/PySyft) - A Python library for secure and private Deep Learning built on PyTorch and TensorFlow.
* [X] [PyGrid](https://github.com/OpenMined/PyGrid/) - Peer-to-peer network of data owners and data scientists who can collectively train AI models using PySyft
* [X] [sktime](https://github.com/alan-turing-institute/sktime) - A unified framework for machine learning with time series
* [X] [OPFython](https://github.com/gugarosa/opfython) - A Python-inspired implementation of the Optimum-Path Forest classifier.
* [X] [Opytimizer](https://github.com/gugarosa/opytimizer) - Python-based meta-heuristic optimization techniques.
* [X] [Gradio](https://github.com/gradio-app/gradio) - A Python library for quickly creating and sharing demos of models. Debug models interactively in your browser, get feedback from collaborators, and generate public links without deploying anything.
* [X] [Hub](https://github.com/activeloopai/Hub) - Fastest unstructured dataset management for TensorFlow/PyTorch. Stream & version-control data. Store even petabyte-scale data in a single numpy-like array on the cloud accessible on any machine. Visit [activeloop.ai](https://activeloop.ai) for more info.
* [X] [Synthia](https://github.com/dmey/synthia) - Multidimensional synthetic data generation in Python.
* [X] [ByteHub](https://github.com/bytehub-ai/bytehub) - An easy-to-use, Python-based feature store. Optimized for time-series data.
* [X] [Backprop](https://github.com/backprop-ai/backprop) - Backprop makes it simple to use, finetune, and deploy state-of-the-art ML models.
* [X] [River](https://github.com/online-ml/river): A framework for general purpose online machine learning.
* [X] [FEDOT](https://github.com/nccr-itmo/FEDOT): An AutoML framework for the automated design of composite modeling pipelines. It can handle classification, regression, and time series forecasting tasks on different types of data (including multi-modal datasets).
* [X] [Sklearn-genetic-opt](https://github.com/rodrigo-arenas/Sklearn-genetic-opt): An AutoML package for hyperparameters tuning using evolutionary algorithms, with built-in callbacks, plotting, remote logging and more.
* [X] [Evidently](https://github.com/evidentlyai/evidently): Interactive reports to analyze machine learning models during validation or production monitoring.
* [X] [Streamlit](https://github.com/streamlit/streamlit): Streamlit is an framework to create beautiful data apps in hours, not weeks.
* [X] [Optuna](https://github.com/optuna/optuna): Optuna is an automatic hyperparameter optimization software framework, particularly designed for machine learning.
* [X] [Deepchecks](https://github.com/deepchecks/deepchecks): Validation & testing of machine learning models and data during model development, deployment, and production. This includes checks and suites related to various types of issues, such as model performance, data integrity, distribution mismatches, and more.
* [X] [Shapash](https://github.com/MAIF/shapash) : Shapash is a Python library that provides several types of visualization that display explicit labels that everyone can understand.
* [X] [Eurybia](https://github.com/MAIF/eurybia): Eurybia monitors data and model drift over time and securizes model deployment with data validation.
* [X] [Colossal-AI](https://github.com/hpcaitech/ColossalAI): An open-source deep learning system for large-scale model training and inference with high efficiency and low cost.
* [X] [dirty_cat](https://github.com/dirty-cat/dirty_cat) - facilitates machine-learning on dirty, non-curated categories. It provides transformers and encoders robust to morphological variants, such as typos.
* [X] [Upgini](https://github.com/upgini/river): Free automated data & feature enrichment library for machine learning - automatically searches through thousands of ready-to-use features from public and community shared data sources and enriches your training dataset with only the accuracy improving features.    
- [X] [pytorch](https://pytorch.org/): The torch package contains data structures for multi-dimensional tensors and defines mathematical operations over these tensors. Additionally, it provides many utilities for efficient serializing of Tensors and arbitrary types, and other useful utilities.  
- [X] [pytorch-transformers](https://github.com/huggingface/transformers): State-of-the-art Natural Language Processing for TensorFlow 2.0 and PyTorch 	
- [X] [finetune](https://github.com/IndicoDataSolutions/finetune): Scikit-learn style model finetuning for NLP  
- [X] [gdal](https://pypi.org/project/GDAL/): GDAL: Geospatial Data Abstraction Library  
- [X] [gensim](https://pypi.org/project/gensim/): Topic modelling, document indexing and similarity retrieval with large corpora.  
- [X] [hungabunga](https://github.com/ypeleg/HungaBunga): HungaBunga: Brute-Force all sklearn models with all parameters using .fit .predict!  
- [X] [implicit](https://github.com/benfred/implicit): Fast Python Collaborative Filtering for Implicit Feedback Datasets  
- [X] [interpret](https://pypi.org/project/interpret/): Fit interpretable models. Explain blackbox machine learning.  - [keras](https://pypi.org/project/Keras/): High-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano.  
- [X] [libffm](https://github.com/ycjuan/libffm): A Library for Field-aware Factorization Machines  
- [X] [libfm](http://www.libfm.org/): Factorization Machine Library  
- [X] [lightfm](https://github.com/lyst/lightfm): A Python implementation of LightFM, a hybrid recommendation algorithm. 
- [X] [lime](https://pypi.org/project/lime/): Local Interpretable Model-Agnostic Explanations for machine learning classifiers  
- [X] [matchzoo](https://matchzoo.readthedocs.io/en/master/): MatchZoo is a toolkit for text matching  
- [X] [matplotlib](https://pypi.org/project/matplotlib/): Matplotlib strives to produce publication quality 2D graphics  - [newspaper](https://pypi.org/project/newspaper/): Simplified python article discovery & extraction.  
- [X] [nlopt](https://pypi.org/project/nlopt/): Library for nonlinear optimization, wrapping many algorithms for global and local, constrained or unconstrained, optimization  
- [X] [nltk](https://pypi.org/project/nltk/): Natural Language Toolkit  
- [X] [opencv](https://pypi.org/project/opencv-python/): Wrapper package for OpenCV python bindings.  
- [X] [pandarallel](https://pypi.org/project/pandarallel/): An easy to use library to speed up computation (by parallelizing on multi CPUs) with pandas.  
- [X] [pandas](https://pypi.org/project/pandas/): Powerful data structures for data analysis, time series, and statistics  
- [X] [pdf2image](https://pypi.org/project/pdf2image/): A wrapper around the pdftoppm and pdftocairo command line tools to convert PDF to a PIL Image list.  
- [X] [X] [pillow](https://pypi.org/project/Pillow/): Python Imaging Library (Fork) 
- [X] [plotly](https://pypi.org/project/plotly/): An open-source, interactive graphing library for Python  
- [X] [prophet](https://pypi.org/project/prophet/): Microframework for analyzing financial markets.  
- [X] [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/): PyMuPDF is a Python binding for MuPDF – a lightweight PDF, XPS, and E-book viewer, renderer, and toolkit, which is maintained and developed by Artifex Software, Inc
- [X] [scikit-image](https://scikit-image.org/): Collection of algorithms for image processing  
- [X] [scikit-learn](https://scikit-learn.org/stable/): Tools for data mining and data analysis and machine learning in Python  
- [X] [scikit-surprise](https://pypi.org/project/scikit-surprise/): Python RecommendatIon System Engine  
- [X] [X] [scrapy](https://scrapy.org/): Framework for extracting the data you need from websites  
- [X] [seaborn](https://seaborn.pydata.org/): Data visualization library based on matplotlib.  
- [X] [selenium](https://selenium-python.readthedocs.io/): Provides a simple API to write functional/acceptance tests using Selenium WebDriver  
- [X] [shap](https://github.com/slundberg/shap): Explain the output of any machine learning model  
- [X] [shutil](https://docs.python.org/3/library/shutil.html): Offers a number of high-level operations on files and collections of files  
- [X] [spacy](https://spacy.io/): Library for advanced Natural Language Processing in Python 
- [X] [sympy](https://www.sympy.org/): Python library for symbolic mathematics  
- [X] [tensorflow](https://www.tensorflow.org/): Core open source library to develop and train ML models  
- [X] [tqdm](https://tqdm.github.io/): Displays progress bar for list iterations
- [X] [xgboost](https://xgboost.readthedocs.io/en/latest/index.html): Distributed gradient boosting library   
- [X] [xlearn](https://github.com/aksnzhy/xlearn): High performance, easy-to-use, and scalable machine learning package 


### Be familar with Computer Vision libraries
- [X] [PCL](https://pointclouds.org/)
    - [X] []()
- [X] [Opencv](https://docs.opencv.org/4.x/)
- [ ] [Scikit-Image](https://github.com/scikit-image/scikit-image)
- [ ] [Scikit-Opt](https://github.com/guofei9987/scikit-opt)
- [ ] [SimpleCV ](http://simplecv.org/
- [ ] [Vigranumpy](https://github.com/ukoethe/vigra)
- [ ] [OpenFace ](https://cmusatyalab.github.io/openface/)
- [ ] [PCV](https://github.com/jesolem/PCV)
- [ ] [face_recognition ](https://github.com/ageitgey/face_recognition)
- [ ] [deepface ](https://github.com/serengil/deepface)
- [ ] [retinaface](https://github.com/serengil/retinaface)
- [ ] [dockerface](https://github.com/natanielruiz/dockerface)
- [ ] [Detectron ](https://github.com/facebookresearch/Detectron)
- [ ] [detectron2](https://github.com/facebookresearch/detectron2)
- [ ] [albumentations ](https://github.com/albumentations-team/albumentations)
- [ ] [pytessarct](https://github.com/madmaze/pytesseract)
- [ ] [imutils ](https://github.com/PyImageSearch/imutils)
- [ ] [PyTorchCV](https://github.com/donnyyou/PyTorchCV)
- [ ] [neural-style-pt](https://github.com/ProGamerGov/neural-style-pt)
- [ ] [Detecto ](https://github.com/alankbi/detecto)
- [ ] [neural-dream](https://github.com/ProGamerGov/neural-dream)
- [ ] [Openpose ](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
- [ ] [Deep High-Resolution-Net](https://github.com/josephmisiti/awesome-machine-learning#python-computer-vision)
- [ ] [dream-creator](https://github.com/ProGamerGov/dream-creator)
- [ ] [Lucent](https://github.com/greentfrapp/lucent)
- [ ] [lightly](https://github.com/lightly-ai/lightly)
- [ ] [Learnergy](https://github.com/gugarosa/learnergy)
- [ ] [OpenVisionAPI ](https://github.com/openvisionapi)
- [ ] [IoT Owl ](https://github.com/Ret2Me/IoT-Owl)
- [ ] [Exadel CompreFace ](https://github.com/exadel-inc/CompreFace)
- [ ] [computer-vision-in-action ](https://github.com/Charmve/computer-vision-in-action)
- [X] [Midjourney: Text-to-Image](https://www.midjourney.com/home/) > Expanding the imaginative powers of the human species
- [X] [DataGradients](https://docs.deci.ai/data-gradients/documentation/feature_configuration.html)

### Be familar with NLP libraries
- [X] [NLTK]()
- [X] [spaCy]()
- [X] [Gensim]()


### MLOPS

- [Docker](https://docs.docker.com/)
- [Kubernetes](https://kubernetes.io/docs/home/)
- [Google PubSub](https://cloud.google.com/pubsub/docs)
- [Bigtable](https://cloud.google.com/bigtable/docs)
- [Spanner](https://cloud.google.com/spanner/docs)
- [Vertica](https://www.vertica.com/docs/9.2.x/HTML/Content/Home.htm)
- [Vitess](https://vitess.io/docs/)
- [Iceberg](https://iceberg.apache.org/docs/1.2.0/)



##  System Design 

- [X] [System Design Using AWS Services](https://www.youtube.com/watch?v=h8NpIop9Lho)
      
    - **Client**
	- **AWS DNS service**
	    - [Amazon Route 53](https://aws.amazon.com/route53/)
    - **AWS Load Balancer**
        - [Elastic Load Balancing](https://aws.amazon.com/elasticloadbalancing/)
    - **AWS API gateway**
        - [Amazon API Gateway](https://aws.amazon.com/api-gateway/)
    - **AWS Proxy Services**

		We can create proxy inside your computing machine like EC2
        - [Be anonymous, create your own proxy server with AWS EC2](https://dev.to/viralsangani/be-anonymous-create-your-own-proxy-server-with-aws-ec2-2k63)

    - **AWS Redundancy and replication**
        
		We can create replication of different services sepereatly.
        - [S3 Replication and Redundancy with Managed Services in AWS](https://medium.com/@gabanox/aws-data-replication-and-redundancy-with-managed-services-2e4d2a0fe98e)
        - [How to meet business data resiliency with Amazon S3 cross-Region replication](https://aws.amazon.com/blogs/publicsector/how-to-meet-business-data-resiliency-s3-cross-region-replication/)
        - [Disaster recovery options in the cloud](https://docs.aws.amazon.com/whitepapers/latest/disaster-recovery-workloads-on-aws/disaster-recovery-options-in-the-cloud.html)
        - [Working with an AWS DMS replication instance](https://docs.aws.amazon.com/dms/latest/userguide/CHAP_ReplicationInstance.html)
        - [Replicating objects](https://docs.aws.amazon.com/AmazonS3/latest/userguide/replication.html)
  
    - **AWS storage services**
		[Object, file, and block storage](https://aws.amazon.com/products/storage/) 

    - **AWS Message queues** : AWS message queue services are a fully managed service that makes it easy to decouple applications and microservices and scale them independently. It provides a reliable, scalable, and cost-effective way to send and receive messages between applications.
        - [SQS]()

    - **Microservices** 
        - [X] [EC2: Run microservices inside EC2](): Autoscaling group help to autoscale services.
        - [X] [AWS Lambda]() : Each microservices backend will be implemented in seperate lamba functions. Lambda scale automatically so ther is no autoscaling group in this case.
        - [X] [Amazon Elastic kubernetes container service]() : Run container inside Amazon elastic contain


### **System design diagram tools**
- [X] [diagrams.net](https://www.diagrams.net/)
    - [Connect diagram.net with github](https://www.diagrams.net/blog/github-support)
	- [single-repository-diagrams](https://www.diagrams.net/blog/single-repository-diagrams)
	- [Edit diagrams directly in GitHub with diagrams.net and github.dev](https://www.diagrams.net/blog/edit-diagrams-with-github-dev)
	- For private repository you can edit diagrm either from vscode extension or add private repository in drawio application github application config
- [X] [lucidchart](https://www.lucidchart.com/blog)
