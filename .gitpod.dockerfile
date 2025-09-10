FROM gitpod/workspace-full

RUN echo "CI version from base"

# Upgrade pip and essential Python packages
USER gitpod
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel virtualenv pipenv

# Setup Heroku CLI
RUN curl https://cli-assets.heroku.com/install.sh | sh

# Add aliases
RUN echo 'alias heroku_config=". $GITPOD_REPO_ROOT/.vscode/heroku_config.sh"' >> ~/.bashrc && \
    echo 'alias python=python3' >> ~/.bashrc && \
    echo 'alias pip=pip3' >> ~/.bashrc && \
    echo 'alias arctictern="python3 $GITPOD_REPO_ROOT/.vscode/arctictern.py"' >> ~/.bashrc && \
    echo 'alias font_fix="python3 $GITPOD_REPO_ROOT/.vscode/font_fix.py"' >> ~/.bashrc && \
    echo 'alias make_url="python3 $GITPOD_REPO_ROOT/.vscode/make_url.py "' >> ~/.bashrc

# Local environment variables
ENV PORT="8080"
ENV IP="0.0.0.0"
