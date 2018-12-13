now=$(date +"%Y-%m-%d %H:%M:%S")

/usr/bin/hadoop/software/hadoop/bin/hadoop fs -rmr  /home/hdp-btime/yujianmin/workplace/tf/output
/usr/bin/hadoop/software/hbox/bin/hbox-submit \
   --app-type "tensorflow" \
   --driver-memory 1024 \
   --driver-cores 1 \
   --files input_from_csv.1.4.py \
   --worker-memory 2048 \
   --worker-cores 1 \
   --worker-gpus 2 \
   --board-enable true \
   --hbox-cmd "python input_from_csv.1.4.py --train_path=./train_data_path --test_file=./test_data --save_path=./model --training_epochs=20 --batch_size=100 --learning_rate=0.5" \
   --input hdfs://path:9000/home/your/workplace/tf/input/MNIST_data_trans2#train_data_path \
   --input hdfs://path:9000/home/your/workplace/tf/input/MNIST_data_trans/test_files/test.data#test_data \
   --output hdfs://path:9000/home/your/workplace/tf/output#model \
   --priority VERY_HIGH \
   --appName "tfdemo_test_single" \
echo "view result"
/usr/bin/hadoop/software/hadoop/bin/hadoop fs -ls  /home/hdp-btime/yujianmin/workplace/tf/output


end=$(date +"%Y-%m-%d %H:%M:%S")
echo 'begin : ' $now
echo 'end   : ' $end
