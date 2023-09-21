FROM python:3.9-slim

RUN python --version

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output/images/automated-petct-lesion-segmentation \
    && chown -R algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip


COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
COPY --chown=algorithm:algorithm dynunet_inference.py /opt/algorithm/
COPY --chown=algorithm:algorithm swinunetr_inference.py /opt/algorithm/


RUN python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN python -m pip install --user -rrequirements.txt

COPY --chown=algorithm:algorithm best_swin_metric_model.pth /opt/algorithm/
COPY --chown=algorithm:algorithm best_dyn_metric_model.pth /opt/algorithm/

COPY --chown=algorithm:algorithm process.py /opt/algorithm/

ENTRYPOINT python -m process $0 $@
