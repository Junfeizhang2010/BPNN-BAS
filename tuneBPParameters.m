function wb = tuneBPParameters(data, neuronCount, tuneMode, opt)
    %% BP神经网络初始参数优化
    % 输入：
    %   data             - 数据集
    %   neuronCount      - 每层神经元个数
    %   tuneMode         - 优化算法选择，'BAS'：天牛须算法
    %                                   ''：算法
    %   ifDraw           - 是否画图
    % 输出：
    %   para             - 列向量，优化算法推荐的BP神经网络权重和偏置
    %                    ★一层隐藏层时结构：
    %                                 ┌──────────────────────────┐
    %           ┌               ┌     隐藏层神经元1                数据特征值1
    %           │          neuronCount 隐藏层神经元2                数据特征值1
    %           │               个        ……
    %           │               └     隐藏层神经元neureCount       数据特征值1
    %           │               ┌     隐藏层神经元1                数据特征值2
    %           │          neuronCount 隐藏层神经元2                数据特征值2
    %           │               个        ……
    %           │               └     隐藏层神经元neureCount       数据特征值2
    %   size(trainX,1)×neuronCount                           ……
    %           个               ┌     隐藏层神经元1                数据特征值size(trainX,1)
    %           │          neuronCount 隐藏层神经元2                数据特征值size(trainX,1)
    %           │               个        ……
    %           └               └     隐藏层神经元neureCount       数据特征值size(trainX,1)
    %
    %           ┌               ┌     数据目标值1                  隐藏层神经元1
    %           │          neuronCount 数据目标值1                  隐藏层神经元2
    %           │               个         ……
    %           │               └     数据目标值1                  隐藏层神经元neureCount
    %           │               ┌     数据目标值2                  隐藏层神经元1
    %           │          neuronCount 数据目标值2                  隐藏层神经元2
    %           │               个         ……
    %           │               └     数据目标值2                  隐藏层神经元neureCount
    %   size(trainY,1)×neuronCount         ……
    %           个               ┌     数据目标值size(trainY,1)     隐藏层神经元1
    %           │          neuronCount 数据目标值size(trainY,1)     隐藏层神经元2
    %           │               个         ……
    %           └               └     数据目标值size(trainY,1)     隐藏层神经元neureCount
    %
    %                            ┌     隐藏层神经元1偏置
    %                       neuronCount 隐藏层神经元2偏置
    %                            个        ……
    %                            └     隐藏层神经元neureCount偏置
    %                            ┌     输出层神经元1偏置
    %                   size(trainY,1)  输出层神经元2偏置
    %                            个        ……
    %                            └     输出层神经元size(trainY,1)偏置
    %                                  └──────────────────────────┘
    %                    ★二层隐藏层时结构：
    %                                 ┌──────────────────────────┐
    %           ┌               ┌     隐藏层1神经元1                数据特征值1
    %           │       neuronCount(1) 隐藏层1神经元2                数据特征值1
    %           │               个        ……
    %           │               └     隐藏层1神经元neureCount(1)    数据特征值1
    %           │               ┌     隐藏层1神经元1                数据特征值2
    %           │       neuronCount(1) 隐藏层1神经元2                数据特征值2
    %           │               个        ……
    %           │               └     隐藏层1神经元neureCount(1)    数据特征值2
    %  size(trainX,1) × neuronCount(1)                         ……
    %           个               ┌     隐藏层1神经元1                数据特征值size(trainX,1)
    %           │       neuronCount(1) 隐藏层1神经元2                数据特征值size(trainX,1)
    %           │               个        ……
    %           └               └     隐藏层1神经元neureCount(1)    数据特征值size(trainX,1)
    %
    %           ┌               ┌     隐藏层2神经元1                隐藏层1神经元1
    %           │       neuronCount(2) 隐藏层2神经元2                隐藏层1神经元2
    %           │               个        ……
    %           │               └     隐藏层2神经元neureCount(2)    隐藏层1神经元neureCount(1)
    %           │               ┌     隐藏层2神经元1                隐藏层1神经元1
    %           │       neuronCount(2) 隐藏层2神经元2                隐藏层1神经元2
    %           │               个        ……
    %           │               └     隐藏层2神经元neureCount(2)    隐藏层1神经元neureCount(1)
    %  size(trainX,1) × neuronCount(2)                         ……
    %           个               ┌     隐藏层2神经元1                隐藏层1神经元1
    %           │       neuronCount(2) 隐藏层2神经元2                隐藏层1神经元2
    %           │               个        ……
    %           └               └     隐藏层2神经元neureCount(2)    隐藏层1神经元neureCount(1)
    %
    %           ┌               ┌     数据目标值1                  隐藏层2神经元1
    %           │          neuronCount 数据目标值1                  隐藏层2神经元2
    %           │               个         ……
    %           │               └     数据目标值1                  隐藏层2神经元neureCount(2)
    %           │               ┌     数据目标值2                  隐藏层2神经元1
    %           │          neuronCount 数据目标值2                  隐藏层2神经元2
    %           │               个         ……
    %           │               └     数据目标值2                  隐藏层2神经元neureCount(2)
    %   size(trainY,1)×neuronCount()         ……
    %           个               ┌     数据目标值size(trainY,1)     隐藏层2神经元1
    %           │          neuronCount 数据目标值size(trainY,1)     隐藏层2神经元2
    %           │               个         ……
    %           └               └     数据目标值size(trainY,1)     隐藏层2神经元neureCount(2)
    %
    %                            ┌     隐藏层1神经元1偏置
    %                    neuronCount(1) 隐藏层1神经元2偏置
    %                            个        ……
    %                            └     隐藏层1神经元neureCount(1)偏置
    %                            ┌     隐藏层2神经元1偏置
    %                    neuronCount(1) 隐藏层2神经元2偏置
    %                            个        ……
    %                            └     隐藏层2神经元neureCount(2)偏置
    %                            ┌     输出层神经元1偏置
    %                   size(trainY,1)  输出层神经元2偏置
    %                            个        ……
    %                            └     输出层神经元size(trainY,1)偏置
    %                                  └──────────────────────────┘
    
    algoMode = 'BP';
    fprintf(['            ●正在进行' algoMode '模型初始参数优化，']);
    switch tuneMode
    %% 天牛须算法
    case 'BAS'
        disp(['使用' tuneMode '算法进行调优...']);
        opt.eta     = 0.9; % 设置变步长，越接近目标越慢
        opt.c       = 5;    % c为步长与两须距离的比例，始终固定，即随着步长变短，两须距离也变短。。。（推荐5）
        opt.step    = 1;    % 设置初始步长，尽可能大，最好与自变量最大长度相当
        opt.n       = 100;   % 最大迭代次数
        switch size(neuronCount, 2)
            case 1
                k = size(data.crossColumn{1}.trainX,1) * neuronCount + neuronCount +...
                    size(data.crossColumn{1}.trainY,1) * neuronCount + size(data.crossColumn{1}.trainY,1);
                % 维度 = 特征值数 * 隐藏层神经元个数 + 隐藏层神经元个数 + 目标值数 * 隐藏层神经元个数 + 目标值数
            case 2
                k = size(data.crossColumn{1}.trainX,1) * neuronCount(1) + neuronCount(1) +...
                    neuronCount(1) * neuronCount(2) + neuronCount(2) +...
                    size(data.crossColumn{1}.trainY,1) * neuronCount(2) + size(data.crossColumn{1}.trainY,1);
                % 维度 = 特征值数 * 隐藏层1神经元个数 + 隐藏层1神经元个数 + 隐藏层1神经元个数 * 隐藏层2神经元个数 +
                %        隐藏层2神经元个数 * 目标值数 + 目标值数
        end
        
        x = rands(k,1);   % x为天牛位置，随机初始化为-1到1的数，列向量[维度×1]
        others.neuronCount = neuronCount;
        [xStore, fBestStore] = BAS(data, 'BPParameter', x, opt, others);
        
        % 得出结论
        [~, bestIndex] = min(fBestStore);
        wb = xStore(2 : end - 1, bestIndex);
        
        
        % 画图
        if ~isfield(opt, 'show') || ~isfield(opt.show, 'isDraw') || opt.show.isDraw
            % 最终结果所在的折线收敛图
            figurePara = setFigurePara(data.dataIndex, data.posOutput);
            figurePara.label.x = 'Iteration';
            figurePara.label.y = figurePara.label.mse;
            figurePara.label.title = ['Convergence line of ' algoMode ' with different parameters while ' tuneMode ' tuning'];
            figurePara.fold = ['savedFigure\' data.dataIndex '\' algoMode '\' tuneMode];
            figurePara.preserve.name = ['使用' tuneMode '算法优化' algoMode '权重和偏置的最终结果所在的收敛迭代图'];
            drawSingleLine(1 : size(fBestStore, 1), fBestStore, figurePara);
        end
        disp(['              ' algoMode '模型初始参数优化完毕。']);
        
    %% 算法2
    case ''
        
    end