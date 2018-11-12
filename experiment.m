function experiment(opt)
    %% ���룺    
    %   opt                     - �㷨����
    %      .data.dataName       - ���ݼ�����
    %           .posOutput      - ���ֵ�����ݼ��е���λ�ã�������posOutput�У�Ĭ��1��
    %           .algoMode       - ģ���㷨
    %           .trainPer       - ѵ����������Ĭ��1��������ѵ����/���Լ���
    %           .crossNum       - ������֤����Ĭ��10��
    %      .show.isDraw         - �Ƿ�ͼ��Ĭ��true��
    %      .tune.time           - �Ż��׶�ʵ�����������ȡ��������Ĭ��10��
    %      .compare.pointCount  - �Ա�δ�Ż�����׶β��Դ������������������ʹ�ý�����֤�Ĳ�ͬ���ݣ�Ĭ��100��
    %      .BP.hLMaxCount       - �Ż��㷨���Ż�BPģ��ʱ���Ե���������Ĭ��4��
    %         .init{i}          - �Ż��㷨���Ż�BPģ��ʱ���Ե�i��ĳ�ʼֵ��������4����Ĭ�ϳ�ʼֵ����������
    %         .BAS.eta          - BAS�Ż��㷨���Ż�BPģ��ʱ��ÿ�ε������������ʣ�Ĭ��.95�������ñ䲽����Խ�ӽ�Ŀ��Խ����
    %             .c            - BAS�Ż��㷨���Ż�BPģ��ʱ���������ı�����Ĭ��5����ʼ�չ̶��������Ų�����̣��������Ҳ���,�Ƽ�5��
    %             .step         - BAS�Ż��㷨���Ż�BPģ��ʱ�ĳ�ʼ������Ĭ��Լ���ʼֵ����Ե�����൱���������ܴ�������Ա�����󳤶��൱��
    %             .n            - BAS�Ż��㷨���Ż�BPģ��ʱ��������������Ĭ��50��
    %      .SVM.init            - init(1)���Ż��㷨�Ż�SVMģ�͵�C���Ż�������ϵ������Ϸ����Ե����������󻮷����̶̳ȣ�
    %                             init(2)���Ż��㷨�Ż�SVMģ�͵�gamma���Ż��ĸ�˹�˴�����������������������̶̳ȣ�
    %          .BAS
    %      .DT.init             - init(1)���Ż��㷨�Ż�DTģ�͵�MinLeafSize��Ҷ�ӽڵ���С������
    %                             init(2)���Ż��㷨�Ż�DTģ�͵�MinParentSize���м�ڵ���С������
    %          .BAS
    %      .RF.init             - init(1)���Ż��㷨�Ż�RFģ�͵�NumTrees����������
    %                             init(2)���Ż��㷨�Ż�RFģ�͵�MinLeafSize��Ҷ�ӽڵ���С������
    %         .BAS
    %      .KNN.init            - init���Ż��㷨�Ż�KNNģ�͵�NumNeighbors������������
    %          .BAS

    %% ����׼��
    disp('��1/3������׼��...');
    % ��������
    data = loadData(opt.data.dataName);
    fprintf('       1/4������������ɣ�');                                    

    % ���ݹ�һ��
    if ~isfield(opt, 'data') || ~isfield(opt.data, 'posOutput')
        data.posOutput = 1; 
    else
        data.posOutput = opt.data.posOutput;
    end
    data = dataNormalize(data);  % ���ݹ�һ����ֻ��X��
    fprintf('2/4�����ݹ�һ����ɣ�');

    % ѵ�������Լ�׼��
    if ~isfield(opt, 'data') || ~isfield(opt.data, 'trainPer')
        data.trainPer = 1; 
    else
        data.trainPer = opt.data.trainPer;
    end
    data = getTrainAndTest(data);
    fprintf('3/4��ѵ�������Լ�����׼����ɣ�');

    % ʮ�۽�����֤����׼��
    if ~isfield(opt, 'data') || ~isfield(opt.data, 'crossNum')
        data.crossNum = 10; 
    else
        data.crossNum = opt.data.crossNum;
    end
    data = getCross(data);
    fprintf('4/4��ʮ�۽�����֤����׼����ɡ�\n');
    disp('  ����׼����ɡ�');

    %% ���ݷ���
    disp('��2/4�����ݷ���...');
    % ������������
    % analysisPartialDependency(data);

    %% ѵ�������
    disp('��3/4��ģ��ѡ��...');
    algoMode = opt.data.algoMode;
    tuneMode = 'BAS';   % ��ţ��
    n = size(algoMode, 2);
    err = ones(1, n) * inf;
    R   = - ones(1, n) * inf;
    gap = ones(n, data.count) * inf;
    if ~isfield(opt, 'isDraw') opt.isDraw = true; end;
    for i = 1 : n
        disp(['       ' num2str(i) '/' num2str(n) '������' algoMode{i} 'ģ��...']);
        [err(i), R(i), gap(i,:), bestIteration{i}] = analysisAlgorithm(data, tuneMode, algoMode{i}, opt);
    end
    save([getCurrentPath(mfilename('fullpath')), 'savedData\' opt.data.dataName '\��ģ�Ͳ������������'], 'err', 'R', 'gap');
    disp('  ģ��ѡ����ɡ�');

    %% ģ������
    disp('��4/4��ģ������...');
    % ��ģ���Ż����̵�������
    analysisTune(data, bestIteration, algoMode);
    % ����ͼ����Ԥ��ֵ��ʵֵ�Ĳ��
    analysisBox(data, gap', algoMode);
    disp('  ģ��������ɡ�');
    disp('����ۣ�');
    disp('  ��ģ�͵�Ԥ��ֵMSE��R��Ԥ�������λ���ֱ�Ϊ��');
    for i = 1 : n
        disp(['  ' stretchString(algoMode{i}, 3) '��MSE = ' num2str(round(err(i), 3))...
            '��R = ' num2str(round(R(i), 3)) '��Ԥ�������λ�� = ' num2str(round(median(gap(i,:)), 3))]);
    end
    [~, indexMse] = min(err);
    disp(['  ������СMSE��ģ��Ϊ��' algoMode{indexMse}]);
    [~, indexR] = max(R);
    disp(['  �������R��ģ��Ϊ��' algoMode{indexR}]);
    [~, indexGap] = min(median(gap'));
    disp(['  ������СԤ�������λ����ģ��Ϊ��' algoMode{indexGap}]);