all: compile demo
compile:
	-chmod u+x ./*.sh
	./compile.sh
demo:
	-chmod u+x ./*.sh
	rm -rf output
	mkdir output
	@echo [DEMO] running the serial version of D-Cube...
	./run_single.sh example_data.txt output 3 geo density 3
	@echo [DEMO] found blocks were saved in the local directory output
	@echo [DEMO] uploading the example data to HDFS
	-hadoop fs -rm example_data.txt
	hadoop fs -put example_data.txt .
	@echo [DEMO] running the Hadoop version of D-Cube...
	./run_hadoop.sh example_data.txt dcube_output 3 geo density 3 4 log
	@echo [DEMO] found blocks were saved in the HDFS directory dcube_output

