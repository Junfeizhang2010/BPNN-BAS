function [para, fBestStoreBest] = tuneNormal(data, tuneMode, algoMode, opt)
    %% һ��ģ�Ͳ����Ż�
    % ���룺
    %   data            - ���ݼ�
    %   tuneMode        - �Ż��㷨ѡ��'BAS'����ţ���㷨
    %   algoMode        - ģ���㷨
    %   opt             - �㷨����
    % �����
    %   para            - �Ż��Ĳ���
    %   fBestStore      - ��������ͼ����
    
    fprintf(['            �����ڽ���' algoMode 'ģ�ͳ�ʼ�����Ż���']);
    
    %% ����
    switch tuneMode
    % ��ţ���㷨
    case 'BAS'
        disp(['ʹ��' tuneMode '�㷨���е���...']);
        switch algoMode
            case 'SVM'
                if ~isfield(opt, 'SVM') || ~isfield(opt.SVM, 'BAS') || ~isfield(opt.SVM.BAS, 'eta'),  opt.SVM.BAS.eta  = 0.95; end; % ���ñ䲽����Խ�ӽ�Ŀ��Խ��
                if ~isfield(opt, 'SVM') || ~isfield(opt.SVM, 'BAS') || ~isfield(opt.SVM.BAS, 'c'),    opt.SVM.BAS.c    = 5;    end; % cΪ�������������ı�����ʼ�չ̶��������Ų�����̣��������Ҳ��̡��������Ƽ�5��
                if ~isfield(opt, 'SVM') || ~isfield(opt.SVM, 'BAS') || ~isfield(opt.SVM.BAS, 'step'), opt.SVM.BAS.step = 2;    end; % ���ó�ʼ�����������ܴ�������Ա�����󳤶��൱
                if ~isfield(opt, 'SVM') || ~isfield(opt.SVM, 'BAS') || ~isfield(opt.SVM.BAS, 'n'),    opt.SVM.BAS.n    = 50;   end; % ����������
                optBAS = opt.SVM.BAS;
                if ~isfield(opt, 'SVM') || ~isfield(opt.SVM, 'init')
                    init = rands(2, 1) * 7 - 3;  % init(1)��C���Ż�������ϵ������Ϸ����Ե����������󻮷����̶̳ȣ�
                                                 % init(2)��gamma���Ż��ĸ�˹�˴�����������������������̶̳ȣ�
                                                 % ��ţλ�ã�������
                else
                    init = opt.SVM.init;
                end
            case 'DT'
                if ~isfield(opt, 'DT') || ~isfield(opt.DT, 'BAS') || ~isfield(opt.DT.BAS, 'eta'),  opt.DT.BAS.eta  = 0.95; end; % ���ñ䲽����Խ�ӽ�Ŀ��Խ��
                if ~isfield(opt, 'DT') || ~isfield(opt.DT, 'BAS') || ~isfield(opt.DT.BAS, 'c'),    opt.DT.BAS.c    = 5;    end; % cΪ�������������ı�����ʼ�չ̶��������Ų�����̣��������Ҳ��̡��������Ƽ�5��
                if ~isfield(opt, 'DT') || ~isfield(opt.DT, 'BAS') || ~isfield(opt.DT.BAS, 'step'), opt.DT.BAS.step = 30;   end; % ���ó�ʼ�����������ܴ�������Ա�����󳤶��൱
                if ~isfield(opt, 'DT') || ~isfield(opt.DT, 'BAS') || ~isfield(opt.DT.BAS, 'n'),    opt.DT.BAS.n    = 50;   end; % ����������
                optBAS = opt.DT.BAS;
                if ~isfield(opt, 'DT') || ~isfield(opt.DT, 'init')
                    init = [30 60]'; % init(1)��MinLeafSize��Ҷ�ӽڵ���С������
                                     % init(2)��MinParentSize���м�ڵ���С������
                                     % �������Ż��ģ�NumVariablesToSample  QuadraticErrorTolerance
                                     % ��ţλ�ã�������
                else
                    init = opt.DT.init;
                end
            case 'RF'
                if ~isfield(opt, 'RF') || ~isfield(opt.RF, 'BAS') || ~isfield(opt.RF.BAS, 'eta'),  opt.RF.BAS.eta  = 0.95; end; % ���ñ䲽����Խ�ӽ�Ŀ��Խ��
                if ~isfield(opt, 'RF') || ~isfield(opt.RF, 'BAS') || ~isfield(opt.RF.BAS, 'c'),    opt.RF.BAS.c    = 5;    end; % cΪ�������������ı�����ʼ�չ̶��������Ų�����̣��������Ҳ��̡��������Ƽ�5��
                if ~isfield(opt, 'RF') || ~isfield(opt.RF, 'BAS') || ~isfield(opt.RF.BAS, 'step'), opt.RF.BAS.step = 50;    end; % ���ó�ʼ�����������ܴ�������Ա�����󳤶��൱
                if ~isfield(opt, 'RF') || ~isfield(opt.RF, 'BAS') || ~isfield(opt.RF.BAS, 'n'),    opt.RF.BAS.n    = 50;   end; % ����������
                optBAS = opt.RF.BAS;
                if ~isfield(opt, 'RF') || ~isfield(opt.RF, 'init')
                    init = [50 50]'; % init(1)��NumTrees����������
                                     % init(2)��MinLeafSize��Ҷ�ӽڵ���С������
                                     % ��ţλ�ã�������
                else
                    init = opt.RF.init;
                end
            case 'KNN'
                if ~isfield(opt, 'KNN') || ~isfield(opt.KNN, 'BAS') || ~isfield(opt.KNN.BAS, 'eta'),  opt.KNN.BAS.eta  = 0.95; end; % ���ñ䲽����Խ�ӽ�Ŀ��Խ��
                if ~isfield(opt, 'KNN') || ~isfield(opt.KNN, 'BAS') || ~isfield(opt.KNN.BAS, 'c'),    opt.KNN.BAS.c    = 5;    end; % cΪ�������������ı�����ʼ�չ̶��������Ų�����̣��������Ҳ��̡��������Ƽ�5��
                if ~isfield(opt, 'KNN') || ~isfield(opt.KNN, 'BAS') || ~isfield(opt.KNN.BAS, 'step'), opt.KNN.BAS.step = 50;    end; % ���ó�ʼ�����������ܴ�������Ա�����󳤶��൱
                if ~isfield(opt, 'KNN') || ~isfield(opt.KNN, 'BAS') || ~isfield(opt.KNN.BAS, 'n'),    opt.KNN.BAS.n    = 50;   end; % ����������
                optBAS = opt.KNN.BAS;
                if ~isfield(opt, 'KNN') || ~isfield(opt.KNN, 'init')
                    init = 100;       % init��NumNeighbors������������
                else
                    init = opt.KNN.init;
                end
            case 'LR'
                if ~isfield(opt, 'LR') || ~isfield(opt.LR, 'BAS') || ~isfield(opt.LR.BAS, 'eta'),  opt.LR.BAS.eta  = 0.95; end; % ���ñ䲽����Խ�ӽ�Ŀ��Խ��
                if ~isfield(opt, 'LR') || ~isfield(opt.LR, 'BAS') || ~isfield(opt.LR.BAS, 'c'),    opt.LR.BAS.c    = 5;    end; % cΪ�������������ı�����ʼ�չ̶��������Ų�����̣��������Ҳ��̡��������Ƽ�5��
                if ~isfield(opt, 'LR') || ~isfield(opt.LR, 'BAS') || ~isfield(opt.LR.BAS, 'step'), opt.LR.BAS.step = 4;    end; % ���ó�ʼ�����������ܴ�������Ա�����󳤶��൱
                if ~isfield(opt, 'LR') || ~isfield(opt.LR, 'BAS') || ~isfield(opt.LR.BAS, 'n'),    opt.LR.BAS.n    = 50;   end; % ����������
                optBAS = opt.LR.BAS;
                if ~isfield(opt, 'LR') || ~isfield(opt.LR, 'init')
                    init  = [4 4]';   % init(1)��������ֵ��ָ����10e-para(1)
                                      % init(2)��ѧϰ���ʱ�����0.0006 + 0.0001 * para(2)
                                      % ��ţλ�ã����������ݵ��ڲ��� )
               	else
                    init = opt.LR.init;
                end
            case 'NB'
                if ~isfield(opt, 'NB') || ~isfield(opt.NB, 'BAS') || ~isfield(opt.NB.BAS, 'eta'),  opt.NB.BAS.eta  = 0.95; end; % ���ñ䲽����Խ�ӽ�Ŀ��Խ��
                if ~isfield(opt, 'NB') || ~isfield(opt.NB, 'BAS') || ~isfield(opt.NB.BAS, 'c'),    opt.NB.BAS.c    = 5;    end; % cΪ�������������ı�����ʼ�չ̶��������Ų�����̣��������Ҳ��̡��������Ƽ�5��
                if ~isfield(opt, 'NB') || ~isfield(opt.NB, 'BAS') || ~isfield(opt.NB.BAS, 'step'), opt.NB.BAS.step = 4;    end; % ���ó�ʼ�����������ܴ�������Ա�����󳤶��൱
                if ~isfield(opt, 'NB') || ~isfield(opt.NB, 'BAS') || ~isfield(opt.NB.BAS, 'n'),    opt.NB.BAS.n    = 50;   end; % ����������
                optBAS = opt.NB.BAS;
                if ~isfield(opt, 'NB') || ~isfield(opt.NB, 'init')
                    init  = []';   % init(1)��������ֵ��ָ����10e-para(1)
                                      % init(2)��ѧϰ���ʱ�����0.0006 + 0.0001 * para(2)
                                      % ��ţλ�ã����������ݵ��ڲ��� )
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
        
        % �ó�����
        [~, bestIndex] = min(fBestStoreBest);
        para = xStore(2 : end - 1, bestIndex);
        
    % �㷨2
    case ''

    end 
    
    fprintf(['              ' algoMode 'ģ�ͽṹ�Ż���ϣ�' tuneMode '�㷨�Ƽ�']);
    switch algoMode
        case 'SVM'
            disp(['C = ' num2str(para(1)) '��gamma = ' num2str(para(2)) '��']);
        case 'DT'
            disp(['Ҷ�ӽڵ��������������С��' num2str(para(1)) '�����м�ڵ��������������С��' num2str(para(2)) '����']);
        case 'RF'
            disp([num2str(para(1)) '������ÿ����Ҷ�ӽڵ���С����' num2str(para(2)) '��������']);
        case 'KNN'
            disp(['ÿ����������' num2str(para) '������������']);
        case 'LR'
            disp(['������ֵ = 10e-' num2str(para(1)) '��ѧϰ���� = ' num2str(0.0006 + 0.0001 * para(2)) '��']);
        case 'NB'

    end
        
    %% ��ͼ
    if ~isfield(opt, 'show') || ~isfield(opt.show, 'isDraw') || opt.show.isDraw
        % �Ż���������ͼ
        figurePara = setFigurePara(data.dataIndex, data.posOutput);
        figurePara.label.x = 'Iteration';
        figurePara.label.y = figurePara.label.mse;
        figurePara.label.title = ['Convergence line of ' algoMode ' with different parameters while' tuneMode 'tuning'];
        figurePara.preserve.name = ['ʹ��' tuneMode '�㷨�Ż���' algoMode 'ģ����������ͼ'];
        figurePara.fold = ['savedFigure\' data.dataIndex '\' algoMode '\' tuneMode];
        figurePara.label.legend = {'Best MSE', 'Current MSE'};
        temp{1} = fBestStoreBest';
        temp{2} = xStore(end,:);
        drawMultipleLine(temp, figurePara);
    end
    disp(['              ' algoMode 'ģ�Ͳ����Ż���ϡ�']);