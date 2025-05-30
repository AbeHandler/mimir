import os
os.environ["MIMIR_DATA_SOURCE"] = "mimirdata"
os.environ["MIMIR_CACHE_PATH"] = "mimrcache"

rule fin:
    input:
        [".snake.analysis", ".snake.copywrite_traps", "olmo_blocked_docs_m1.csv", "olmo_blocked_docs_m0.csv"]
    shell:
        "rm -f gurobi.log"

rule by_publisher:
    output:
        ".snake.by_publisher"
    shell:
        "MIMIR_DATA_SOURCE=mimirdata MIMIR_CACHE_PATH=mimrcache CUDA_VISIBLE_DEVICES=0,1 conda run --live-stream -n mimir python run.py --config configs/olmo_by_publisher_real.json && echo 'done' > {output}"

rule blocked_docs_m1:
    output:
        proof=".snake.blocked_docs.m1"
    shell:
        "MIMIR_DATA_SOURCE=mimirdata MIMIR_CACHE_PATH=mimrcache CUDA_VISIBLE_DEVICES=0,1 conda run --live-stream -n mimir python run.py --config configs/olmo_blocked_docs_m1.json && echo 'done' > {output.proof}"

rule blocked_docs_m0:
    output:
        proof=".snake.blocked_docs.m0"
    shell:
        "MIMIR_DATA_SOURCE=mimirdata MIMIR_CACHE_PATH=mimrcache CUDA_VISIBLE_DEVICES=0,1 conda run --live-stream -n mimir python run.py --config configs/olmo_blocked_docs_m0.json && echo 'done' > {output.proof}"

rule post_process_blocked_docs:
    input:
        proof=[".snake.blocked_docs.m0", ".snake.blocked_docs.m1"]
    output:
        "olmo_blocked_docs_m1.csv",
        "olmo_blocked_docs_m0.csv",
        "mimir.E1.csv"
    shell:
        """
        conda run --live-stream -n analysis python build_output.py --config olmo_blocked_docs_m0
        conda run --live-stream -n analysis python build_output.py --config olmo_blocked_docs_m1
        conda run --live-stream -n analysis python merge_output.py
        """

rule copywrite_traps:
    output:
        ".snake.copywrite_traps"
    shell:
        "MIMIR_DATA_SOURCE=mimirdata MIMIR_CACHE_PATH=mimrcache conda run --live-stream -n mimir python run.py --config configs/copywrite_traps.json && echo 'done' > {output}"


rule analysis:
    input:
        ".snake.by_publisher"
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
