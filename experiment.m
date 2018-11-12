function experiment(opt)
    %% 输入：    
    %   opt                     - 算法参数
    %      .data.dataName       - 数据集名称
    %           .posOutput      - 输出值在数据集中的列位置（倒数第posOutput列，默认1）
    %           .algoMode       - 模型算法
    %           .trainPer       - 训练集比例（默认1，即不分训练集/测试集）
    %           .crossNum       - 交叉验证数（默认10）
    %      .show.isDraw         - 是否画图（默认true）
    %      .tune.time           - 优化阶段实验次数，用于取最好情况（默认10）
    %      .compare.pointCount  - 对比未优化情况阶段测试次数（点数），会均匀使用交叉验证的不同数据（默认100）
    %      .BP.hLMaxCount       - 优化算法在优化BP模型时测试的最大层数（默认4）
    %         .init{i}          - 优化算法在优化BP模型时测试第i层的初始值（不大于4层有默认初始值，列向量）
    %         .BAS.eta          - BAS优化算法在优化BP模型时的每次迭代步长减少率（默认.95）（设置变步长，越接近目标越慢）
    %             .c            - BAS优化算法在优化BP模型时的两须距离的比例（默认5）（始终固定，即随着步长变短，两须距离也变短,推荐5）
    %             .step         - BAS优化算法在优化BP模型时的初始步长（默认约与初始值到边缘距离相当）（尽可能大，最好与自变量最大长度相当）
    %             .n            - BAS优化算法在优化BP模型时的最大迭代次数（默认50）
    %      .SVM.init            - init(1)：优化算法优化SVM模型的C，优化的正则化系数（拟合非线性的能力，错误划分容忍程度）
    %                             init(2)：优化算法优化SVM模型的gamma，优化的高斯核带宽（拟合线性能力，噪声容忍程度）
    %          .BAS
    %      .DT.init             - init(1)：优化算法优化DT模型的MinLeafSize，叶子节点最小样本数
    %                             init(2)：优化算法优化DT模型的MinParentSize，中间节点最小样本数
    %          .BAS
    %      .RF.init             - init(1)：优化算法优化RF模型的NumTrees，树的数量
    %                             init(2)：优化算法优化RF模型的MinLeafSize，叶子节点最小样本数
    %         .BAS
    %      .KNN.init            - init：优化算法优化KNN模型的NumNeighbors，样本近邻数
    %          .BAS

    %% 数据准备
    disp('★1/3：数据准备...');
    % 载入数据
    data = loadData(opt.data.dataName);
    fprintf('       1/4：数据载入完成；');                                    

    % 数据归一化
    if ~isfield(opt, 'data') || ~isfield(opt.data, 'posOutput')
        data.posOutput = 1; 
    else
        data.posOutput = opt.data.posOutput;
    end
    data = dataNormalize(data);  % 数据归一化（只有X）
    fprintf('2/4：数据归一化完成；');

    % 训练集测试集准备
    if ~isfield(opt, 'data') || ~isfield(opt.data, 'trainPer')
        data.trainPer = 1; 
    else
        data.trainPer = opt.data.trainPer;
    end
    data = getTrainAndTest(data);
    fprintf('3/4：训练集测试集数据准备完成；');

    % 十折交叉验证数据准备
    if ~isfield(opt, 'data') || ~isfield(opt.data, 'crossNum')
        data.crossNum = 10; 
    else
        data.crossNum = opt.data.crossNum;
    end
    data = getCross(data);
    fprintf('4/4：十折交叉验证数据准备完成。\n');
    disp('  数据准备完成。');

    %% 数据分析
    disp('★2/4：数据分析...');
    % 部分依赖分析
    % analysisPartialDependency(data);

    %% 训练与测试
    disp('★3/4：模型选择...');
    algoMode = opt.data.algoMode;
    tuneMode = 'BAS';   % 天牛须
    n = size(algoMode, 2);
    err = ones(1, n) * inf;
    R   = - ones(1, n) * inf;
    gap = ones(n, data.count) * inf;
    if ~isfield(opt, 'isDraw') opt.isDraw = true; end;
    for i = 1 : n
        disp(['       ' num2str(i) '/' num2str(n) '：测试' algoMode{i} '模型...']);
        [err(i), R(i), gap(i,:), bestIteration{i}] = analysisAlgorithm(data, tuneMode, algoMode{i}, opt);
    end
    save([getCurrentPath(mfilename('fullpath')), 'savedData\' opt.data.dataName '\各模型测试情况的数据'], 'err', 'R', 'gap');
    disp('  模型选择完成。');

    %% 模型评估
    disp('★4/4：模型评估...');
    % 各模型优化过程迭代分析
    analysisTune(data, bestIteration, algoMode);
    % 箱型图分析预测值真实值的差别
    analysisBox(data, gap', algoMode);
    disp('  模型评估完成。');
    disp('★结论：');
    disp('  各模型的预测值MSE、R和预测误差中位数分别为：');
    for i = 1 : n
        disp(['  ' stretchString(algoMode{i}, 3) '：MSE = ' num2str(round(err(i), 3))...
            '，R = ' num2str(round(R(i), 3)) '，预测误差中位数 = ' num2str(round(median(gap(i,:)), 3))]);
    end
    [~, indexMse] = min(err);
    disp(['  具有最小MSE的模型为：' algoMode{indexMse}]);
    [~, indexR] = max(R);
    disp(['  具有最大R的模型为：' algoMode{indexR}]);
    [~, indexGap] = min(median(gap'));
    disp(['  具有最小预测误差中位数的模型为：' algoMode{indexGap}]);