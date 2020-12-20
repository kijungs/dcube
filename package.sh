rm DCube-2.0.tar.gz
rm -rf DCube-2.0
mkdir DCube-2.0
cp -R ./{run_*.sh,compile.sh,package.sh,src,./output,Makefile,README.txt,*.jar,example_data.txt,user_guide.pdf} ./DCube-2.0
tar cvzf DCube-2.0.tar.gz --exclude='._*' ./DCube-2.0
rm -rf DCube-2.0
echo done.
