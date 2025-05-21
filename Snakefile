import os
os.environ["MIMIR_DATA_SOURCE"] = "mimirdata"
os.environ["MIMIR_CACHE_PATH"] = "mimrcache"

rule all:
    input:
        [".snake.analysis", ".snake.copywrite_traps"]

rule run_mimir:
    output:
        ".snake.mimrran"
    shell:
        "MIMIR_DATA_SOURCE=mimirdata MIMIR_CACHE_PATH=mimrcache CUDA_VISIBLE_DEVICES=0,1 conda run --live-stream -n mimir python run.py --config configs/olmo_by_publisher_real.json && echo 'done' > {output}"


rule copywrite_traps:
    output:
        ".snake.copywrite_traps"
    shell:
        "MIMIR_DATA_SOURCE=mimirdata MIMIR_CACHE_PATH=mimrcache conda run --live-stream -n mimir python run.py --config configs/copywrite_traps.json && echo 'done' > {output}"


rule analysis:
    input:
        ".snake.mimrran"
    output:
        ".snake.analysis"
    shell:
        r"""
        # 1. Create the env (if it doesn’t already exist)
        conda create --name analysis python=3.9 -y

        # 2. “Hook” conda into this shell session
        eval "$(conda shell.bash hook)"

        # 3. Now you can activate
        conda activate analysis

        # 4. Install your requirements
        pip install -r configs/analysis_requirements.txt

        # 5. Do the analysis
        conda run --live-stream -n analysis python scripts/process_olmo_by_publisher.py

        # 6. Mark as completed
        touch {output}
        """
