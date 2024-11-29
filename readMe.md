This guide will help you set up the Python environment and install the necessary libraries to run this project.

1. Create a Python Virtual Environment

We use Python 3.11 for TensorFlow compatibility
```bash
python3.11 -m venv aa1-env
```

2. Activate the Virtual Environment
```bash
source aa1-env/bin/activate
```

3. Install the Required Libraries
```bash
pip install -r requirements.txt
```

4. Deactivate the Virtual Environment
```bash
deactivate
```

5. Build the Docker image based on the docker file
```bash
docker build .
```

6. Run the Docker we build
```bash
docker run -p 3000:3000 (id)
```