import os
os.environ["MIMIR_DATA_SOURCE"] = "mimirdata"
os.environ["MIMIR_CACHE_PATH"] = "mimrcache"

rule all:
    input:
        ".snake.mimrran"

rule run_mimir:
    output:
        ".snake.mimrran"
    shell:
        "MIMIR_DATA_SOURCE=mimirdata MIMIR_CACHE_PATH=mimrcache conda run --live-stream -n mimir python run.py --config configs/olmo_by_publisher_dev.json && echo done > {output}"

