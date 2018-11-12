function [para, fBestStoreBest] = tuneNormal(data, tuneMode, algoMode, opt)
    %% 一般模型参数优化
    % 输入：
    %   data            - 数据集
    %   tuneMode        - 优化算法选择，'BAS'：天牛须算法
    %   algoMode        - 模型算法
    %   opt             - 算法参数
    % 输出：
    %   para            - 优化的参数
    %   fBestStore      - 迭代收敛图数据
    
    fprintf(['            ●正在进行' algoMode '模型初始参数优化，']);
    
    %% 计算
    switch tuneMode
    % 天牛须算法
    case 'BAS'
        disp(['使用' tuneMode '算法进行调优...']);
        switch algoMode
            case 'SVM'
                if ~isfield(opt, 'SVM') || ~isfield(opt.SVM, 'BAS') || ~isfield(opt.SVM.BAS, 'eta'),  opt.SVM.BAS.eta  = 0.95; end; % 设置变步长，越接近目标越慢
                if ~isfield(opt, 'SVM') || ~isfield(opt.SVM, 'BAS') || ~isfield(opt.SVM.BAS, 'c'),    opt.SVM.BAS.c    = 5;    end; % c为步长与两须距离的比例，始终固定，即随着步长变短，两须距离也变短。。。（推荐5）
                if ~isfield(opt, 'SVM') || ~isfield(opt.SVM, 'BAS') || ~isfield(opt.SVM.BAS, 'step'), opt.SVM.BAS.step = 2;    end; % 设置初始步长，尽可能大，最好与自变量最大长度相当
                if ~isfield(opt, 'SVM') || ~isfield(opt.SVM, 'BAS') || ~isfield(opt.SVM.BAS, 'n'),    opt.SVM.BAS.n    = 50;   end; % 最大迭代次数
                optBAS = opt.SVM.BAS;
                if ~isfield(opt, 'SVM') || ~isfield(opt.SVM, 'init')
                    init = rands(2, 1) * 7 - 3;  % init(1)：C，优化的正则化系数（拟合非线性的能力，错误划分容忍程度）
                                                 % init(2)：gamma，优化的高斯核带宽（拟合线性能力，噪声容忍程度）
                                                 % 天牛位置，列向量
                else
                    init = opt.SVM.init;
                end
            case 'DT'
                if ~isfield(opt, 'DT') || ~isfield(opt.DT, 'BAS') || ~isfield(opt.DT.BAS, 'eta'),  opt.DT.BAS.eta  = 0.95; end; % 设置变步长，越接近目标越慢
                if ~isfield(opt, 'DT') || ~isfield(opt.DT, 'BAS') || ~isfield(opt.DT.BAS, 'c'),    opt.DT.BAS.c    = 5;    end; % c为步长与两须距离的比例，始终固定，即随着步长变短，两须距离也变短。。。（推荐5）
                if ~isfield(opt, 'DT') || ~isfield(opt.DT, 'BAS') || ~isfield(opt.DT.BAS, 'step'), opt.DT.BAS.step = 30;   end; % 设置初始步长，尽可能大，最好与自变量最大长度相当
                if ~isfield(opt, 'DT') || ~isfield(opt.DT, 'BAS') || ~isfield(opt.DT.BAS, 'n'),    opt.DT.BAS.n    = 50;   end; % 最大迭代次数
                optBAS = opt.DT.BAS;
                if ~isfield(opt, 'DT') || ~isfield(opt.DT, 'init')
                    init = [30 60]'; % init(1)：MinLeafSize，叶子节点最小样本数
                                     % init(2)：MinParentSize，中间节点最小样本数
                                     % 其它可优化的：NumVariablesToSample  QuadraticErrorTolerance
                                     % 天牛位置，列向量
                else
                    init = opt.DT.init;
                end
            case 'RF'
                if ~isfield(opt, 'RF') || ~isfield(opt.RF, 'BAS') || ~isfield(opt.RF.BAS, 'eta'),  opt.RF.BAS.eta  = 0.95; end; % 设置变步长，越接近目标越慢
                if ~isfield(opt, 'RF') || ~isfield(opt.RF, 'BAS') || ~isfield(opt.RF.BAS, 'c'),    opt.RF.BAS.c    = 5;    end; % c为步长与两须距离的比例，始终固定，即随着步长变短，两须距离也变短。。。（推荐5）
                if ~isfield(opt, 'RF') || ~isfield(opt.RF, 'BAS') || ~isfield(opt.RF.BAS, 'step'), opt.RF.BAS.step = 50;    end; % 设置初始步长，尽可能大，最好与自变量最大长度相当
                if ~isfield(opt, 'RF') || ~isfield(opt.RF, 'BAS') || ~isfield(opt.RF.BAS, 'n'),    opt.RF.BAS.n    = 50;   end; % 最大迭代次数
                optBAS = opt.RF.BAS;
                if ~isfield(opt, 'RF') || ~isfield(opt.RF, 'init')
                    init = [50 50]'; % init(1)：NumTrees，树的数量
                                     % init(2)：MinLeafSize，叶子节点最小样本数
                                     % 天牛位置，列向量
                else
                    init = opt.RF.init;
                end
            case 'KNN'
                if ~isfield(opt, 'KNN') || ~isfield(opt.KNN, 'BAS') || ~isfield(opt.KNN.BAS, 'eta'),  opt.KNN.BAS.eta  = 0.95; end; % 设置变步长，越接近目标越慢
                if ~isfield(opt, 'KNN') || ~isfield(opt.KNN, 'BAS') || ~isfield(opt.KNN.BAS, 'c'),    opt.KNN.BAS.c    = 5;    end; % c为步长与两须距离的比例，始终固定，即随着步长变短，两须距离也变短。。。（推荐5）
                if ~isfield(opt, 'KNN') || ~isfield(opt.KNN, 'BAS') || ~isfield(opt.KNN.BAS, 'step'), opt.KNN.BAS.step = 50;    end; % 设置初始步长，尽可能大，最好与自变量最大长度相当
                if ~isfield(opt, 'KNN') || ~isfield(opt.KNN, 'BAS') || ~isfield(opt.KNN.BAS, 'n'),    opt.KNN.BAS.n    = 50;   end; % 最大迭代次数
                optBAS = opt.KNN.BAS;
                if ~isfield(opt, 'KNN') || ~isfield(opt.KNN, 'init')
                    init = 100;       % init：NumNeighbors，样本近邻数
                else
                    init = opt.KNN.init;
                end
            case 'LR'
                if ~isfield(opt, 'LR') || ~isfield(opt.LR, 'BAS') || ~isfield(opt.LR.BAS, 'eta'),  opt.LR.BAS.eta  = 0.95; end; % 设置变步长，越接近目标越慢
                if ~isfield(opt, 'LR') || ~isfield(opt.LR, 'BAS') || ~isfield(opt.LR.BAS, 'c'),    opt.LR.BAS.c    = 5;    end; % c为步长与两须距离的比例，始终固定，即随着步长变短，两须距离也变短。。。（推荐5）
                if ~isfield(opt, 'LR') || ~isfield(opt.LR, 'BAS') || ~isfield(opt.LR.BAS, 'step'), opt.LR.BAS.step = 4;    end; % 设置初始步长，尽可能大，最好与自变量最大长度相当
                if ~isfield(opt, 'LR') || ~isfield(opt.LR, 'BAS') || ~isfield(opt.LR.BAS, 'n'),    opt.LR.BAS.n    = 50;   end; % 最大迭代次数
                optBAS = opt.LR.BAS;
                if ~isfield(opt, 'LR') || ~isfield(opt.LR, 'init')
                    init  = [4 4]';   % init(1)：收敛阈值负指数，10e-para(1)
                                      % init(2)：学习速率倍数，0.0006 + 0.0001 * para(2)
                                      % 天牛位置，列向量（暂调节不了 )
               	else
                    init = opt.LR.init;
                end
            case 'NB'
                if ~isfield(opt, 'NB') || ~isfield(opt.NB, 'BAS') || ~isfield(opt.NB.BAS, 'eta'),  opt.NB.BAS.eta  = 0.95; end; % 设置变步长，越接近目标越慢
                if ~isfield(opt, 'NB') || ~isfield(opt.NB, 'BAS') || ~isfield(opt.NB.BAS, 'c'),    opt.NB.BAS.c    = 5;    end; % c为步长与两须距离的比例，始终固定，即随着步长变短，两须距离也变短。。。（推荐5）
                if ~isfield(opt, 'NB') || ~isfield(opt.NB, 'BAS') || ~isfield(opt.NB.BAS, 'step'), opt.NB.BAS.step = 4;    end; % 设置初始步长，尽可能大，最好与自变量最大长度相当
                if ~isfield(opt, 'NB') || ~isfield(opt.NB, 'BAS') || ~isfield(opt.NB.BAS, 'n'),    opt.NB.BAS.n    = 50;   end; % 最大迭代次数
                optBAS = opt.NB.BAS;
                if ~isfield(opt, 'NB') || ~isfield(opt.NB, 'init')
                    init  = []';   % init(1)：收敛阈值负指数，10e-para(1)
                                      % init(2)：学习速率倍数，0.0006 + 0.0001 * para(2)
                                      % 天牛位置，列向量（暂调节不了 )
               	else
                    init = opt.NB.init;
                end
        end
        if ~isfield(opt, 'tune') || ~isfield(opt.tune, 'time')
            optBAS.time = 10; 
        else
            optBAS.time = opt.tune.time;
        end
        [xStore, fBestStoreBest] = BAS(data, algoMode, init, optBAS, []);
        
        % 得出结论
        [~, bestIndex] = min(fBestStoreBest);
        para = xStore(2 : end - 1, bestIndex);
        
    % 算法2
    case ''

    end 
    
    fprintf(['              ' algoMode '模型结构优化完毕，' tuneMode '算法推荐']);
    switch algoMode
        case 'SVM'
            disp(['C = ' num2str(para(1)) '，gamma = ' num2str(para(2)) '。']);
        case 'DT'
            disp(['叶子节点包含的样本不能小于' num2str(para(1)) '个，中间节点包含的样本不能小于' num2str(para(2)) '个。']);
        case 'RF'
            disp([num2str(para(1)) '棵树，每棵树叶子节点最小包含' num2str(para(2)) '个样本。']);
        case 'KNN'
            disp(['每个样本考虑' num2str(para) '个近邻样本。']);
        case 'LR'
            disp(['收敛阈值 = 10e-' num2str(para(1)) '，学习速率 = ' num2str(0.0006 + 0.0001 * para(2)) '。']);
        case 'NB'

    end
        
    %% 画图
    if ~isfield(opt, 'show') || ~isfield(opt.show, 'isDraw') || opt.show.isDraw
        % 优化折线收敛图
        figurePara = setFigurePara(data.dataIndex, data.posOutput);
        figurePara.label.x = 'Iteration';
        figurePara.label.y = figurePara.label.mse;
        figurePara.label.title = ['Convergence line of ' algoMode ' with different parameters while' tuneMode 'tuning'];
        figurePara.preserve.name = ['使用' tuneMode '算法优化的' algoMode '模型收敛迭代图'];
        figurePara.fold = ['savedFigure\' data.dataIndex '\' algoMode '\' tuneMode];
        figurePara.label.legend = {'Best MSE', 'Current MSE'};
        temp{1} = fBestStoreBest';
        temp{2} = xStore(end,:);
        drawMultipleLine(temp, figurePara);
    end
    disp(['              ' algoMode '模型参数优化完毕。']);