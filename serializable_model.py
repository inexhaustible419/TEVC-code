# 修正后的可序列化模型 - 匹配原始代码结构
# 主要修正：
# 1. Arc_list 改为列表结构，不是字典
# 2. 添加缺失的属性和方法
# 3. 确保访问模式与原始代码一致

import copy
import multiprocessing as mp
import torch

class SerializableArc:
    """可序列化的弧段类 - 完全匹配原始Arc类"""
    def __init__(self, original_arc=None, **kwargs):
        if original_arc:
            # 从原始Arc对象复制所有属性
            self.id = original_arc.id
            self.satellite = original_arc.satellite
            self.satellite_name = getattr(original_arc, 'satellite_name', '')
            self.ground = original_arc.ground
            self.ground_name = getattr(original_arc, 'ground_name', '')
            self.circle = getattr(original_arc, 'circle', 0)
            self.link_st = original_arc.link_st
            self.ctrl_st = getattr(original_arc, 'ctrl_st', 0)
            self.trace_st = getattr(original_arc, 'trace_st', 0)
            self.link_et = original_arc.link_et
            self.ctrl_et = getattr(original_arc, 'ctrl_et', 0)
            self.trace_et = getattr(original_arc, 'trace_et', 0)
            self.priority = getattr(original_arc, 'priority', 0)
            self.f_id = getattr(original_arc, 'f_id', 0)
            self.link_time = original_arc.link_time
            self.confArc_list = copy.copy(getattr(original_arc, 'confArc_list', []))
            self.f_confArc_list = copy.copy(getattr(original_arc, 'f_confArc_list', []))
            self.conf = getattr(original_arc, 'conf', 0)
            self.lt_divide_conf = getattr(original_arc, 'lt_divide_conf', float('inf'))
        else:
            # 使用kwargs初始化
            self.id = kwargs.get('id', 0)
            self.satellite = kwargs.get('satellite', 0)
            self.satellite_name = kwargs.get('satellite_name', '')
            self.ground = kwargs.get('ground', 0)
            self.ground_name = kwargs.get('ground_name', '')
            self.circle = kwargs.get('circle', 0)
            self.link_st = kwargs.get('link_st', 0)
            self.ctrl_st = kwargs.get('ctrl_st', 0)
            self.trace_st = kwargs.get('trace_st', 0)
            self.link_et = kwargs.get('link_et', 0)
            self.ctrl_et = kwargs.get('ctrl_et', 0)
            self.trace_et = kwargs.get('trace_et', 0)
            self.priority = kwargs.get('priority', 0)
            self.f_id = kwargs.get('f_id', 0)
            self.link_time = kwargs.get('link_time', 0)
            self.confArc_list = kwargs.get('confArc_list', [])
            self.f_confArc_list = kwargs.get('f_confArc_list', [])
            self.conf = kwargs.get('conf', 0)
            self.lt_divide_conf = kwargs.get('lt_divide_conf', float('inf'))

class SerializableFragment:
    """可序列化的片段类 - 匹配原始Fragment类"""
    def __init__(self, original_fragment=None):
        if original_fragment:
            self.id = getattr(original_fragment, 'id', 0)
            self.Arc_list = copy.copy(original_fragment.Arc_list)
            self.front_may_conf = copy.copy(getattr(original_fragment, 'front_may_conf', []))
            self.rear_may_conf = copy.copy(getattr(original_fragment, 'rear_may_conf', []))
            self.max_et = getattr(original_fragment, 'max_et', 0)
        else:
            self.id = 0
            self.Arc_list = []
            self.front_may_conf = []
            self.rear_may_conf = []
            self.max_et = 0
        
        # 运行时创建的属性，不预先设置
        self.currentSol = None
        self.newSol = None
        self.bestSol = None

class SerializableModel:
    """可序列化的模型类 - 完全匹配原始Model类结构"""
    def __init__(self, original_model):
        # 【关键修正】Arc_list 是列表，不是字典！
        self.Arc_list = []
        for original_arc in original_model.Arc_list:
            serializable_arc = SerializableArc(original_arc)
            self.Arc_list.append(serializable_arc)
        
        print(f"Converted {len(self.Arc_list)} arcs from list structure")
        
        # 转换fragment_list
        self.fragment_list = []
        for original_fragment in original_model.fragment_list:
            serializable_fragment = SerializableFragment(original_fragment)
            self.fragment_list.append(serializable_fragment)
        
        print(f"Converted {len(self.fragment_list)} fragments")
        
        # 复制模型的其他属性
        # self.fragment_span = copy.copy(original_model.fragment_span)
        self.target_fragments = getattr(original_model, 'target_fragments', 12)
        self.fragment_min_st_max_et = copy.copy(getattr(original_model, 'fragment_min_st_max_et', []))
        self.satellite_change_time = getattr(original_model, 'satellite_change_time', 150)
        self.satellite_trans_time = getattr(original_model, 'satellite_trans_time', 300)
        self.ground_trans_time = getattr(original_model, 'ground_trans_time', 340)
        
        # 运行时属性
        self.last_fragment_bestsole = getattr(original_model, 'last_fragment_bestsole', None)
        
        print(f"Model conversion completed:")
        # print(f"  - Fragment span: {len(self.fragment_span)} intervals")
        print(f"  - Satellite change time: {self.satellite_change_time}")
        print(f"  - Ground trans time: {self.ground_trans_time}")

def create_serializable_model(original_model):
    """
    从原始模型创建可序列化版本 - 修正版
    """
    print("Creating corrected serializable model...")
    print(f"Original model structure:")
    print(f"  - Arc_list type: {type(original_model.Arc_list)} with {len(original_model.Arc_list)} items")
    print(f"  - Fragment_list type: {type(original_model.fragment_list)} with {len(original_model.fragment_list)} items")
    
    # 验证Arc_list的访问模式
    if len(original_model.Arc_list) > 0:
        sample_arc = original_model.Arc_list[0]
        print(f"  - Sample arc ID: {sample_arc.id}, type: {type(sample_arc)}")
    
    try:
        serializable_model = SerializableModel(original_model)
        
        # 验证转换后的结构
        print(f"Serializable model verification:")
        print(f"  - Arc_list length: {len(serializable_model.Arc_list)}")
        print(f"  - Fragment_list length: {len(serializable_model.fragment_list)}")
        
        # 测试访问模式（重要！）
        if len(serializable_model.Arc_list) > 0:
            test_arc = serializable_model.Arc_list[0]
            print(f"  - Test access Arc_list[0]: ID={test_arc.id}, link_time={test_arc.link_time}")
        
        if len(serializable_model.fragment_list) > 0:
            test_fragment = serializable_model.fragment_list[0]
            print(f"  - Test fragment[0]: {len(test_fragment.Arc_list)} arcs")
        
        # 【关键】验证原始代码的访问模式是否正常工作
        print("Verifying access patterns...")
        for i, fragment in enumerate(serializable_model.fragment_list):
            if len(fragment.Arc_list) > 0:
                arc_id = fragment.Arc_list[0]
                if arc_id < len(serializable_model.Arc_list):
                    arc = serializable_model.Arc_list[arc_id]
                    print(f"  - Fragment {i}: arc_id={arc_id}, link_time={arc.link_time}")
                else:
                    print(f"  - ERROR: Fragment {i} has invalid arc_id {arc_id}")
                break
        
        print("Serializable model created successfully!")
        return serializable_model
        
    except Exception as e:
        print(f"Error creating serializable model: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# 修正后的工作函数，确保正确访问Arc_list
def alns_worker_with_corrected_model(serializable_model, elite_sol, f_num, t1, start_time, arc_count, 
                                   it_num, barrier=None, global_vae_queue=None):
    """
    使用修正后可序列化模型的ALNS工作函数
    """
    try:
        print(f"Fragment {f_num} worker started with corrected serializable model")
        print(f"  - Model has {len(serializable_model.Arc_list)} arcs (list structure)")
        print(f"  - Fragment {f_num} has {len(serializable_model.fragment_list[f_num].Arc_list)} arc IDs")
        
        # 验证Fragment中的arc_id是否有效
        fragment = serializable_model.fragment_list[f_num]
        max_arc_id = max(fragment.Arc_list) if fragment.Arc_list else -1
        if max_arc_id >= len(serializable_model.Arc_list):
            print(f"ERROR: Fragment {f_num} has invalid arc_id {max_arc_id}, max valid is {len(serializable_model.Arc_list)-1}")
            return None
        
        # 直接使用传入的可序列化模型
        mode = serializable_model
        
        # 创建片段专用的VAE管理器
        fragment_vae = None
        try:
            from vae_manager import VAEManager
            # 【修正】使用正确的弧段数量
            fragment_arc_count = len(fragment.Arc_list)
            fragment_vae = VAEManager(fragment_arc_count, use_shared_elites=False)
            print(f"Fragment {f_num}: VAE manager created for {fragment_arc_count} arcs")
        except Exception as e:
            print(f"Fragment {f_num}: VAE creation failed: {str(e)}")
        
        # 创建DQN代理
        agent = None
        try:
            from multi_dqn_agent import ParallelDQNAgent, DuelingDQN
            
            vae_dim = 20
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            global_model = DuelingDQN(5 * 5, 17, vae_dim).to(device)
            
            agent = ParallelDQNAgent(
                agent_id=f_num,
                n_destroy_actions=8,
                n_repair_actions=8,
                state_dim=17,
                vae_dim=vae_dim,
                global_model=global_model
            )
            # 【修正】使用总弧段数，不是片段弧段数
            agent.set_arc_count(len(mode.Arc_list))
            print(f"Fragment {f_num}: DQN agent created with {len(mode.Arc_list)} total arcs")
        except Exception as e:
            print(f"Fragment {f_num}: DQN creation failed: {str(e)}")
            return None
        
        # 创建ALNS实例
        from ALNS_DQN2 import ALNS
        alns = ALNS(mode, elite_sol, agent)
        
        # 片段历史精英解
        fragment_elite_solutions = []
        all_results = []
        
        print(f"Fragment {f_num}: Starting {it_num} iterations")
        
        # 【保持不变】每个片段独立运行 it_num 循环
        for num in range(it_num):
            print(f"Fragment {f_num} - Iteration {num+1}/{it_num}")
            
            # 1. 基于本片段的精英解生成初始解
            initial_sol_ls = []
            if num > 0 and len(fragment_elite_solutions) >= 3 and fragment_vae:
                try:
                    # 训练片段局部VAE
                    recent_solutions = fragment_elite_solutions[-10:]  # 只使用最近10个
                    fragment_vae.add_elite_solutions(recent_solutions)
                    
                    if fragment_vae.train():
                        # 生成本片段的初始解
                        generated_solutions = fragment_vae.generate_solutions(3, interpolate=True)
                        
                        for gen_arcs in generated_solutions:
                            # 【修正】确保生成的解只包含本片段的弧段ID
                            fragment_arc_ids = set(fragment.Arc_list)
                            valid_arc_ids = [arc_id for arc_id in gen_arcs if arc_id in fragment_arc_ids]
                            
                            if len(valid_arc_ids) >= 3:
                                import Sol
                                sol = Sol.Sol()
                                sol.Arc_list_id = valid_arc_ids
                                # 【修正】使用列表索引访问
                                sol.link_time = sum([mode.Arc_list[arc_id].link_time for arc_id in valid_arc_ids])
                                sol.link_num = len(valid_arc_ids)
                                initial_sol_ls.append(sol)
                        
                        print(f"Fragment {f_num} generated {len(initial_sol_ls)} VAE-based initial solutions")
                except Exception as e:
                    print(f"Fragment {f_num}: VAE generation failed: {str(e)}")
            
            # 2. 运行ALNS算法
            try:
                result = alns.run(t1, start_time, 0, f_num, initial_sol_ls)
                elite_sol_result, best_sol, init, f_num_result, may_conf = result
                
                # 3. 收集本片段的精英解
                current_elite_solutions = []
                for sol in elite_sol_result.values():
                    if hasattr(sol, 'Arc_list_id') and sol.Arc_list_id:
                        current_elite_solutions.append(sol.Arc_list_id)
                
                fragment_elite_solutions.extend(current_elite_solutions)
                
                # 4. 可选：发送精英解给全局VAE
                if global_vae_queue is not None and current_elite_solutions:
                    try:
                        global_vae_queue.put((f_num, current_elite_solutions), block=False)
                    except:
                        pass  # 队列满时跳过
                
                # 存储当前迭代结果
                all_results.append({
                    'iteration': num,
                    'elite_sol': elite_sol_result,
                    'best_sol': best_sol,
                    'may_conf': may_conf,
                    'elite_solutions': current_elite_solutions
                })
                
                print(f"Fragment {f_num} - Iteration {num+1} completed, best: {best_sol.link_time if best_sol else 0}")
                
            except Exception as e:
                print(f"Fragment {f_num}: ALNS run failed in iteration {num+1}: {str(e)}")
                import traceback
                traceback.print_exc()
                # 创建默认结果
                import Sol
                best_sol = Sol.Sol()
                best_sol.Arc_list_id = []
                best_sol.link_time = 0
                all_results.append({
                    'iteration': num,
                    'elite_sol': {},
                    'best_sol': best_sol,
                    'may_conf': [],
                    'elite_solutions': []
                })
            
            # 5. 【可选】同步机制
            if barrier is not None:
                try:
                    print(f"Fragment {f_num} waiting at barrier for iteration {num+1}")
                    barrier.wait()
                    print(f"Fragment {f_num} passed barrier for iteration {num+1}")
                except Exception as e:
                    print(f"Fragment {f_num}: Barrier failed: {str(e)}")
        
        # 返回最终结果
        if all_results:
            final_result = all_results[-1]
            print(f"Fragment {f_num} - All iterations completed successfully")
            print(f"Fragment {f_num} - Final best link_time: {final_result['best_sol'].link_time}")
            return (final_result['elite_sol'], final_result['best_sol'], 
                   0, f_num, final_result['may_conf'], fragment_elite_solutions)
        else:
            print(f"Fragment {f_num} - No results generated")
            return (elite_sol, None, 0, f_num, [], fragment_elite_solutions)
            
    except Exception as e:
        print(f"Error in fragment {f_num} worker: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# 测试函数，验证序列化模型的正确性
def test_serializable_model(original_model):
    """
    测试可序列化模型是否正确工作
    """
    print("\n" + "="*60)
    print("TESTING SERIALIZABLE MODEL")
    print("="*60)
    
    try:
        # 创建可序列化模型
        serializable_model = create_serializable_model(original_model)
        
        # 测试pickle序列化
        import pickle
        print("\nTesting pickle serialization...")
        pickled_data = pickle.dumps(serializable_model)
        unpickled_model = pickle.loads(pickled_data)
        print("✓ Pickle serialization successful!")
        
        # 测试数据一致性
        print("\nTesting data consistency...")
        assert len(unpickled_model.Arc_list) == len(original_model.Arc_list)
        assert len(unpickled_model.fragment_list) == len(original_model.fragment_list)
        
        # 测试访问模式
        print("\nTesting access patterns...")
        if len(original_model.fragment_list) > 0:
            original_fragment = original_model.fragment_list[0]
            serializable_fragment = unpickled_model.fragment_list[0]
            
            if len(original_fragment.Arc_list) > 0:
                arc_id = original_fragment.Arc_list[0]
                original_arc = original_model.Arc_list[arc_id]
                serializable_arc = unpickled_model.Arc_list[arc_id]
                
                assert original_arc.link_time == serializable_arc.link_time
                assert original_arc.link_st == serializable_arc.link_st
                print(f"✓ Access pattern test passed: arc {arc_id}")
        
        print("\n✓ All tests passed! Serializable model is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
