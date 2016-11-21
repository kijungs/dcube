if [ -z "$HADOOP_CORE" ]; then

    if [ -z "$HADOOP_HOME" ]; then
        HADOOP_HOME=$(echo `which hadoop` | sed 's~bin/hadoop~~g')
	    	echo "The hadoop path is detected! ($HADOOP_HOME)"
    fi

	if [ -n "$HADOOP_HOME" ]; then
    HADOOP_CORE="$HADOOP_HOME/`ls $HADOOP_HOME/ | grep core | grep jar`"
		echo "The path of hadoop core library ($HADOOP_CORE)"
	fi
fi

echo "compiling java sources..."
rm -rf class
mkdir class

if [ -z "$HADOOP_CORE" ]; then
    echo "Failed to find the hadoop core library (jar file) in $HADOOP_HOME."
	echo "Please set the environment variable \$HADOOP_CORE to the path of the hadoop core library."
	javac -cp ./hadoop-core-1.0.3.jar -d class $(find ./src -name *.java)
else
    javac -cp $HADOOP_CORE -d class $(find ./src -name *.java)
fi

echo "make jar archive..."
cd class
jar cf DCube-1.0.jar ./
rm ../DCube-1.0.jar
mv DCube-1.0.jar ../
cd ..
rm -rf class

echo "done."
