import os
os.environ["MIMIR_DATA_SOURCE"] = "mimirdata"
os.environ["MIMIR_CACHE_PATH"] = "mimrcache"

rule fin:
    input:
        [".snake.analysis", ".snake.copywrite_traps", "olmo_blocked_docs_m1.csv", "olmo_blocked_docs_m0.csv"]
    shell:
        "rm -f gurobi.log"

rule run_olmo_by_publisher_real:
    input:
        ".snake.conda"
    output:
        ".snake.by_publisher"
    shell:
        "MIMIR_DATA_SOURCE=mimirdata MIMIR_CACHE_PATH=mimrcache CUDA_VISIBLE_DEVICES=0,1 conda run --live-stream -n mimir python run.py --config configs/olmo_by_publisher_real.json && echo 'done' > {output}"

rule blocked_docs_m1:
    input:
        ".snake.conda"
    output:
        proof=".snake.blocked_docs.m1"
    shell:
        "MIMIR_DATA_SOURCE=mimirdata MIMIR_CACHE_PATH=mimrcache CUDA_VISIBLE_DEVICES=0,1 conda run --live-stream -n mimir python run.py --config configs/olmo_blocked_docs_m1.json && echo 'done' > {output.proof}"

rule blocked_docs_m0:
    input:
        ".snake.conda"
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
    input:
        ".snake.conda"
    output:
        ".snake.copywrite_traps"
    shell:
        "MIMIR_DATA_SOURCE=mimirdata MIMIR_CACHE_PATH=mimrcache conda run --live-stream -n mimir python run.py --config configs/copywrite_traps.json && echo 'done' > {output}"

rule reset_cache:
    output:
        ".snake.reset_cache"
    shell:
        """
        ./wipe.sh
        """

rule init_conda:
    input:
        ".snake.reset_cache"
    output:
        proof=".snake.conda"
    shell:
        """
        eval "$(conda shell.bash hook)"

        conda remove --name analysis --all
        conda create --name analysis python=3.9 -y

        eval "$(conda shell.bash hook)"

        conda activate analysis

        pip install -r configs/analysis_requirements.txt

        conda deactivate

        conda remove --name mimir --all
        conda create --name mimir python=3.9 -y
        conda activate mimir
        pip install -r configs/requirements_w_versions.txt
        """


rule in_sample_by_publisher_analysis:
    input:
        ".snake.by_publisher"
    output:
        proof=".snake.analysis"
    shell:
        r"""
        conda run --live-stream -n analysis python scripts/process_olmo_by_publisher.py

        # 6. Mark as completed
        touch {output.proof}
        """
