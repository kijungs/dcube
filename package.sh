rm DCube-1.0.tar.gz
rm -rf DCube-1.0
mkdir DCube-1.0
cp -R ./{run_*.sh,compile.sh,package.sh,src,./output,Makefile,README.txt,*.jar,example_data.txt,user_guide.pdf} ./DCube-1.0
tar cvzf DCube-1.0.tar.gz --exclude='._*' ./DCube-1.0
rm -rf DCube-1.0
echo done.
