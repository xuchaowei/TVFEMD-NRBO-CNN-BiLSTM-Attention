function fitness = fical(x,Lp_train,t_train,ps_output,T_train,f_);
%%  从主函数中获取训练数据
% 参数只有x，其他是已知的，x就是3个参数的上下界，每次传一个
    Lp_train = evalin('base', 'Lp_train');
    t_train = evalin('base', 't_train');
    ps_output = evalin('base', 'ps_output');
    T_train = evalin('base', 'T_train');
     f_ = evalin('base', 'f_');
    
best_hd  = round(x(1, 2)); % 最佳隐藏层节点数
best_lr= x(1, 1);% 最佳初始学习率
best_l2 = x(1, 3);% 最佳L2正则化系数
%%  建立模型
lgraph = layerGraph();                                                 % 建立空白网络结构


%在深度学习中，输入层的尺寸通常表示为 [height width channels]，所以是[f_, 1, 1]
% 其中 height 是输入图像的高度，width 是宽度，channels 是通道数。
% 对于灰度图像而言，通常只有一个通道，因此 channels 的值为 1。对于时间序列的话，宽度的值为1，所以为[f_, 1, 1]
%对于输入尺寸为 [65 1 1] 的图像数据，sequenceFoldingLayer 会将其转换为一个矩阵，以便后续的卷积层可以处理。
tempLayers = [
    %65*1 1张
    sequenceInputLayer([f_, 1, 1], "Name", "sequence")                 % 建立输入层，输入数据结构为[f_, 1, 1]-就是13*5=65个值
    sequenceFoldingLayer("Name", "seqfold")];                          % 建立序列折叠层 序列折叠层用于将输入序列数据转换成适合卷积层处理的格式。
lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中
% 参考示意图https://blog.51cto.com/u_16099347/10357725
%增加卷积核数量可能会增加网络的表征能力，使网络能够学习更复杂的特征。但是，如果卷积核数量过多，
% 可能会导致过拟合，特别是在数据集较小的情况下。因此，需要根据具体情况进行权衡和选择。
%卷积核的通道数应与输入层的通道数相同，而输出层的通道数则由卷积核的个数决定
%下一层需要多少个featuremap，就计算出多少个卷积核
%有几个卷积核,就会输出几个Feature map，有32个feature图
%65-3+1=63  32张，这32个神经元/卷积核是不同值的
tempLayers = convolution2dLayer([3, 1], 32, "Name", "conv_1");         % 卷积层 卷积核[3, 1-1与上面的1相一致] 步长[1, 1] 通道数 32也就是卷积核数量32
lgraph = addLayers(lgraph,tempLayers);                                 % 将上述网络结构加入空白结构中
 
%为了提升网络的非线性能力，以提高网络的表达能力。每个卷积层后都会跟一个激活层,ReLU 激活函数不改变数据的形状。
tempLayers = [
    reluLayer("Name", "relu_1")  %不改变尺寸加深非线性学习 % 激活层
     %63-3+1=61  64张
    convolution2dLayer([3, 1], 64, "Name", "conv_2")                   % 卷积层 卷积核[3, 1] 步长[1, 1] 通道数 64
    reluLayer("Name", "relu_2")];                                      % 激活层
lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中


%%代替全连接层，可接受任意尺寸的图像 https://blog.csdn.net/SmartDemo/article/details/123889624
%全连接层其实就是一个权重矩阵-全连接层将池化平均值之后的featuremap进行拼接组成新的，参数很多
tempLayers = [
    % 求61*1的64张中每个的均值，然后得到64的 1*1向量 
    globalAveragePooling2dLayer("Name", "gapool")                      % 全局平均池化层
    %16个神经元；每个神经元权重不同，与64个点相计算，得到一个值，总共得到16个1*1
    fullyConnectedLayer(16, "Name", "fc_2")                            % SE注意力机制，通道数的1 / 4
    reluLayer("Name", "relu_3")                                        % 激活层
    %64个神经元；每个神经元权重不同，与16个点相计算，得到一个值，总共得到64个1*1---返回
    fullyConnectedLayer(64, "Name", "fc_3")                            % SE注意力机制，数目和通道数相同--
    %将每个元素的取值范围限制在 0 到 1 之间，表示相应神经元的激活程度。，每个神经元的激活程度
    sigmoidLayer("Name", "sigmoid")];                                  % 激活层
lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中

%点乘注意力
tempLayers = multiplicationLayer(2, "Name", "multiplication");         % 点乘的注意力
lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中

tempLayers = [
    sequenceUnfoldingLayer("Name", "sequnfold")                        % 建立序列反折叠层
    flattenLayer("Name", "flatten")                                    % 网络铺平层
    bilstmLayer(best_hd, "Name", "bilstm", "OutputMode", "last")                 % BiLSTM层
    fullyConnectedLayer(1, "Name", "fc")                               % 全连接层
    regressionLayer("Name", "regressionoutput")];                      % 回归层
lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中

lgraph = connectLayers(lgraph, "seqfold/out", "conv_1");               % 折叠层输出 连接 卷积层输入;
lgraph = connectLayers(lgraph, "seqfold/miniBatchSize", "sequnfold/miniBatchSize"); 
                                                                       % 折叠层输出 连接 反折叠层输入  
lgraph = connectLayers(lgraph, "conv_1", "relu_1");                    % 卷积层输出 链接 激活层
lgraph = connectLayers(lgraph, "conv_1", "gapool");                    % 卷积层输出 链接 全局平均池化
lgraph = connectLayers(lgraph, "relu_2", "multiplication/in2");        % 激活层输出 链接 相乘层
lgraph = connectLayers(lgraph, "sigmoid", "multiplication/in1");       % 全连接输出 链接 相乘层
lgraph = connectLayers(lgraph, "multiplication", "sequnfold/in");      % 点乘输出

%%  参数设置
options = trainingOptions('adam', ...      % Adam 梯度下降算法
    'MaxEpochs', 100, ...                 % 最大迭代次数
    'InitialLearnRate', best_lr, ...          % 初始学习率为0.01
    'LearnRateSchedule', 'piecewise', ...  % 学习率下降
    'LearnRateDropFactor', 0.1, ...        % 学习率下降因子 0.5
     'L2Regularization',best_l2,...
    'LearnRateDropPeriod', 50, ...        % 经过700次训练后 学习率为 0.01 * 0.1
    'Shuffle', 'every-epoch', ...          % 每次训练打乱数据集
    'Plots', 'none', ...      % 画出曲线
    'Verbose', false);


%% 训练混合网络
net = trainNetwork(Lp_train, t_train, lgraph, options);
%% 训练集误差评价
t_sim1 = predict(net, Lp_train);
T_sim1 = mapminmax('reverse', t_sim1', ps_output);
%%  计算适应度
fitness = sqrt(sum((T_sim1 - T_train).^2)./length(T_sim1));
disp(['初始 Fitness = ' num2str(fitness)]);

end