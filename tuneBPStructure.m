function [neuronCount, fBestCompared] = tuneBPStructure(data, tuneMode, opt)
    %% BP神经网络结构优化
    % 输入：
    %   data             - 数据集
    %   tuneMode         - 优化算法选择，'BAS'：天牛须算法
    %   opt              - 算法参数
    % 输出：
    %   neuronCount      - 行向量，优化算法推荐的每层神经元个数
    %   fBestCompared    - 迭代收敛图数据
    
    algoMode = 'BP';
    fprintf(['            ●正在进行' algoMode '模型结构优化，']);
    if ~isfield(opt, 'BP') || ~isfield(opt.BP, 'hiddenLevelMax'), opt.BP.hLMaxCount = 3; end; % 试验的最大层数
    
    %% 计算
    switch tuneMode
    % 天牛须算法
    case 'BAS'
        disp(['使用' tuneMode '算法进行调优...']);
        if ~isfield(opt, 'BP') || ~isfield(opt.BP, 'BAS') || ~isfield(opt.BP.BAS, 'eta'),  opt.BP.BAS.eta  = 0.95; end; % 设置变步长，越接近目标越慢
        if ~isfield(opt, 'BP') || ~isfield(opt.BP, 'BAS') || ~isfield(opt.BP.BAS, 'c'),    opt.BP.BAS.c    = 5;    end; % c为步长与两须距离的比例，始终固定，即随着步长变短，两须距离也变短。。。（推荐5）
        if ~isfield(opt, 'BP') || ~isfield(opt.BP, 'BAS') || ~isfield(opt.BP.BAS, 'step'), opt.BP.BAS.step = 10;   end; % 设置初始步长，尽可能大，最好与自变量最大长度相当
        if ~isfield(opt, 'BP') || ~isfield(opt.BP, 'BAS') || ~isfield(opt.BP.BAS, 'n'),    opt.BP.BAS.n    = 50;   end; % 最大迭代次数

        % 计算1-hLMaxCount层网络的情况
        if ~isfield(opt, 'BP') || ~isfield(opt.BP, 'init')
            for k = 1 : opt.BP.hLMaxCount
                if k == 1
                    opt.BP.init{k} = 30;
                elseif k == 2
                    opt.BP.init{k} = [20 10]';
                elseif k == 3
                    opt.BP.init{k} = [20 10 10]';
                elseif k == 4
                    opt.BP.init{k} = [10 10 10 10]';
                else
                    opt.BP.init{k} = randperm(40,k)'; 
                end
            end
        end
        for k = 1 : opt.BP.hLMaxCount
            disp(['                  正在测试网络层数为' num2str(k) '/' num2str(opt.BP.hLMaxCount) '的情况...']);
            
            % 用天牛须算法计算不同层数下的最优神经元个数
            if ~isfield(opt, 'tune') || ~isfield(opt.tune, 'time')
                opt.BP.BAS.time = 10; 
            else
                opt.BP.BAS.time = opt.tune.time;
            end
            crossTime = size(data.crossColumn, 2);
            [xStore{k}, fBestStore{k}] = BAS(data, 'BPStructure', opt.BP.init{k}, opt.BP.BAS, []);
            [fBestBest(k), bestIndex(k)] = min(fBestStore{k});
            xBest{k} = xStore{k}(2 : end - 1, bestIndex(k));
        end
        
        % 得出结论
        [fBestBestForAll, hiddenLevelCount] = min(fBestBest);
        neuronCount = xBest{hiddenLevelCount}';
        disp(['              ' algoMode '模型结构优化完毕，天牛须算法推荐'...
            num2str(hiddenLevelCount) '个隐藏层，每层神经元分别为' num2str(neuronCount) '。']);

    % 算法2
    case ''
            
    end
    
    %% 画图
    if ~isfield(opt, 'show') || ~isfield(opt.show, 'isDraw') || opt.show.isDraw
        fBestCompared = fBestStore{hiddenLevelCount};
        n = size(fBestCompared, 1);
        % 最终结果所在的折线收敛图
        figurePara = setFigurePara(data.dataIndex, data.posOutput);
        figurePara.label.x = 'Iteration';
        figurePara.label.y = figurePara.label.mse;
        if hiddenLevelCount == 1
            figurePara.label.title = ['Convergence line of ' algoMode ' with ' num2str(hiddenLevelCount)...
                ' hidden level and different number of neurons while' tuneMode 'tuning'];
        else
            figurePara.label.title = ['Convergence line of ' algoMode ' with ' num2str(hiddenLevelCount)...
                ' hidden levels and different number of neurons while' tuneMode 'tuning'];
        end
        figurePara.fold = ['savedFigure\' data.dataIndex '\' algoMode '\' tuneMode];
        figurePara.preserve.name = [data.dataIndex '使用' tuneMode '算法优化' algoMode '模型结构的最终结果所在的收敛迭代图'];
        drawSingleLine(1 : n, fBestCompared, figurePara);

        % 所有情况的目标函数值图
        figurePara = setFigurePara(data.dataIndex, data.posOutput);
        figurePara.label.x = 'Iteration';
        figurePara.label.y = figurePara.label.mse;
        figurePara.label.title = ['Convergence line of ' algoMode ' with different number of hidden levels while ' tuneMode ' tuning'];
        figurePara.fold = ['savedFigure\' data.dataIndex '\' algoMode '\' tuneMode];
        figurePara.preserve.name = ['使用' tuneMode '算法优化' algoMode '模型结构各层迭代情况'];
        drawBPLevelsTune(fBestStore, figurePara);
    end