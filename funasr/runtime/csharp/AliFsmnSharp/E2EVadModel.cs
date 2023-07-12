using System.Diagnostics;
using AliFsmnSharp.Model;

namespace AliFsmnSharp;

internal enum VadStateMachine {
    kVadInStateStartPointNotDetected = 1,
    kVadInStateInSpeechSegment = 2,
    kVadInStateEndPointDetected = 3
}

internal enum VadDetectMode {
    kVadSingleUtteranceDetectMode = 0,
    kVadMutipleUtteranceDetectMode = 1
}

internal class E2EVadModel {
    private readonly VadPostConfEntity _vad_opts = new();
    private readonly WindowDetector _windows_detector = new();
    private bool _is_final;
    private int _data_buf_start_frame;
    private int _frm_cnt;
    private int _latest_confirmed_speech_frame;
    private int _lastest_confirmed_silence_frame = -1;
    private int _continous_silence_frame_count;
    private int _vad_state_machine = (int)VadStateMachine.kVadInStateStartPointNotDetected;
    private int _confirmed_start_frame = -1;
    private int _confirmed_end_frame = -1;
    private int _number_end_time_detected;
    private int _sil_frame;
    private int[] _sil_pdf_ids = new int[0];
    private double _noise_average_decibel = -100.0D;
    private bool _pre_end_silence_detected;
    private bool _next_seg = true;

    private List<E2EVadSpeechBufWithDoaEntity> _output_data_buf;
    private int _output_data_buf_offset;
    private List<E2EVadFrameProbEntity> _frame_probs = new();
    private int _max_end_sil_frame_cnt_thresh = 800 - 150;
    private float _speech_noise_thres = 0.6F;
    private float[,,] _scores;
    private int _idx_pre_chunk;
    private bool _max_time_out;
    private List<double> _decibel = new();
    private int _data_buf_size;
    private int _data_buf_all_size;

    public E2EVadModel(VadPostConfEntity vadPostConfEntity) {
        _vad_opts = vadPostConfEntity;
        _windows_detector = new WindowDetector(_vad_opts.window_size_ms,
            _vad_opts.sil_to_speech_time_thres,
            _vad_opts.speech_to_sil_time_thres,
            _vad_opts.frame_in_ms);
        AllResetDetection();
    }

    private void AllResetDetection() {
        _is_final = false;
        _data_buf_start_frame = 0;
        _frm_cnt = 0;
        _latest_confirmed_speech_frame = 0;
        _lastest_confirmed_silence_frame = -1;
        _continous_silence_frame_count = 0;
        _vad_state_machine = (int)VadStateMachine.kVadInStateStartPointNotDetected;
        _confirmed_start_frame = -1;
        _confirmed_end_frame = -1;
        _number_end_time_detected = 0;
        _sil_frame = 0;
        _sil_pdf_ids = _vad_opts.sil_pdf_ids;
        _noise_average_decibel = -100.0F;
        _pre_end_silence_detected = false;
        _next_seg = true;

        _output_data_buf = new List<E2EVadSpeechBufWithDoaEntity>();
        _output_data_buf_offset = 0;
        _frame_probs = new List<E2EVadFrameProbEntity>();
        _max_end_sil_frame_cnt_thresh = _vad_opts.max_end_silence_time - _vad_opts.speech_to_sil_time_thres;
        _speech_noise_thres = _vad_opts.speech_noise_thres;
        _scores = null;
        _idx_pre_chunk = 0;
        _max_time_out = false;
        _decibel = new List<double>();
        _data_buf_size = 0;
        _data_buf_all_size = 0;
        ResetDetection();
    }

    private void ResetDetection() {
        _continous_silence_frame_count = 0;
        _latest_confirmed_speech_frame = 0;
        _lastest_confirmed_silence_frame = -1;
        _confirmed_start_frame = -1;
        _confirmed_end_frame = -1;
        _vad_state_machine = (int)VadStateMachine.kVadInStateStartPointNotDetected;
        _windows_detector.Reset();
        _sil_frame = 0;
        _frame_probs = new List<E2EVadFrameProbEntity>();
    }

    private void ComputeDecibel(float[] waveform) {
        var frame_sample_length = _vad_opts.frame_length_ms * _vad_opts.sample_rate / 1000;
        var frame_shift_length = _vad_opts.frame_in_ms * _vad_opts.sample_rate / 1000;
        if (_data_buf_all_size == 0) {
            _data_buf_all_size = waveform.Length;
            _data_buf_size = _data_buf_all_size;
        } else {
            _data_buf_all_size += waveform.Length;
        }

        for (var offset = 0; offset < waveform.Length - frame_sample_length + 1; offset += frame_shift_length) {
            var _waveform_chunk = new float[frame_sample_length];
            Array.Copy(waveform, offset, _waveform_chunk, 0, _waveform_chunk.Length);
            var _waveform_chunk_pow = _waveform_chunk.Select(x => (float)Math.Pow(x, 2)).ToArray();
            _decibel.Add(
                10 * Math.Log10(
                    _waveform_chunk_pow.Sum() + 0.000001
                )
            );
        }
    }

    private void ComputeScores(float[,,] scores) {
        _vad_opts.nn_eval_block_size = scores.GetLength(1);
        _frm_cnt += scores.GetLength(1);
        _scores = scores;
    }

    private void PopDataBufTillFrame(int frame_idx) // need check again
    {
        while (_data_buf_start_frame < frame_idx)
            if (_data_buf_size >= _vad_opts.frame_in_ms * _vad_opts.sample_rate / 1000) {
                _data_buf_start_frame += 1;
                _data_buf_size = _data_buf_all_size - _data_buf_start_frame *
                    (_vad_opts.frame_in_ms * _vad_opts.sample_rate / 1000);
            }
    }

    private void PopDataToOutputBuf(int start_frm, int frm_cnt, bool first_frm_is_start_point,
        bool last_frm_is_end_point, bool end_point_is_sent_end) {
        PopDataBufTillFrame(start_frm);
        var expected_sample_number = frm_cnt * _vad_opts.sample_rate * _vad_opts.frame_in_ms / 1000;
        if (last_frm_is_end_point) {
            var extra_sample = Math.Max(0,
                _vad_opts.frame_length_ms * _vad_opts.sample_rate / 1000 -
                _vad_opts.sample_rate * _vad_opts.frame_in_ms / 1000);
            expected_sample_number += extra_sample;
        }

        if (end_point_is_sent_end) expected_sample_number = Math.Max(expected_sample_number, _data_buf_size);

        if (_data_buf_size < expected_sample_number) Console.WriteLine("error in calling pop data_buf\n");

        if (_output_data_buf.Count == 0 || first_frm_is_start_point) {
            _output_data_buf.Add(new E2EVadSpeechBufWithDoaEntity());
            _output_data_buf.Last().Reset();
            _output_data_buf.Last().start_ms = start_frm * _vad_opts.frame_in_ms;
            _output_data_buf.Last().end_ms = _output_data_buf.Last().start_ms;
            _output_data_buf.Last().doa = 0;
        }

        var cur_seg = _output_data_buf.Last();
        if (cur_seg.end_ms != start_frm * _vad_opts.frame_in_ms) Console.WriteLine("warning\n");

        var out_pos = cur_seg.buffer.Length; // cur_seg.buff现在没做任何操作
        var data_to_pop = 0;
        if (end_point_is_sent_end)
            data_to_pop = expected_sample_number;
        else
            data_to_pop = frm_cnt * _vad_opts.frame_in_ms * _vad_opts.sample_rate / 1000;

        if (data_to_pop > _data_buf_size) {
            Console.WriteLine("VAD data_to_pop is bigger than _data_buf_size!!!\n");
            data_to_pop = _data_buf_size;
            expected_sample_number = _data_buf_size;
        }


        cur_seg.doa = 0;
        for (var sample_cpy_out = 0; sample_cpy_out < data_to_pop; sample_cpy_out++) out_pos += 1;

        for (var sample_cpy_out = data_to_pop; sample_cpy_out < expected_sample_number; sample_cpy_out++) out_pos += 1;

        if (cur_seg.end_ms != start_frm * _vad_opts.frame_in_ms)
            Console.WriteLine("Something wrong with the VAD algorithm\n");

        _data_buf_start_frame += frm_cnt;
        cur_seg.end_ms = (start_frm + frm_cnt) * _vad_opts.frame_in_ms;
        if (first_frm_is_start_point) cur_seg.contain_seg_start_point = true;

        if (last_frm_is_end_point) cur_seg.contain_seg_end_point = true;
    }

    private void OnSilenceDetected(int valid_frame) {
        _lastest_confirmed_silence_frame = valid_frame;
        if (_vad_state_machine == (int)VadStateMachine.kVadInStateStartPointNotDetected)
            PopDataBufTillFrame(valid_frame);
    }

    private void OnVoiceDetected(int valid_frame) {
        _latest_confirmed_speech_frame = valid_frame;
        PopDataToOutputBuf(valid_frame, 1, false, false, false);
    }

    private void OnVoiceStart(int start_frame, bool fake_result = false) {
        if (_vad_opts.do_start_point_detection) {
            //do nothing
        }

        if (_confirmed_start_frame != -1)
            Console.WriteLine("not reset vad properly\n");
        else
            _confirmed_start_frame = start_frame;

        if (!fake_result || _vad_state_machine == (int)VadStateMachine.kVadInStateStartPointNotDetected)
            PopDataToOutputBuf(_confirmed_start_frame, 1, true, false, false);
    }

    private void OnVoiceEnd(int end_frame, bool fake_result, bool is_last_frame) {
        for (var t = _latest_confirmed_speech_frame + 1; t < end_frame; t++) OnVoiceDetected(t);

        if (_vad_opts.do_end_point_detection) {
            //do nothing
        }

        if (_confirmed_end_frame != -1)
            Console.WriteLine("not reset vad properly\n");
        else
            _confirmed_end_frame = end_frame;

        if (!fake_result) {
            _sil_frame = 0;
            PopDataToOutputBuf(_confirmed_end_frame, 1, false, true, is_last_frame);
        }

        _number_end_time_detected += 1;
    }

    private void MaybeOnVoiceEndIfLastFrame(bool is_final_frame, int cur_frm_idx) {
        if (is_final_frame) {
            OnVoiceEnd(cur_frm_idx, false, true);
            _vad_state_machine = (int)VadStateMachine.kVadInStateEndPointDetected;
        }
    }

    private int GetLatency() {
        return LatencyFrmNumAtStartPoint() * _vad_opts.frame_in_ms;
    }

    private int LatencyFrmNumAtStartPoint() {
        var vad_latency = _windows_detector.GetWinSize();
        if (_vad_opts.do_extend != 0) vad_latency += _vad_opts.lookback_time_start_point / _vad_opts.frame_in_ms;

        return vad_latency;
    }

    private FrameState GetFrameState(int t) {
        var frame_state = FrameState.kFrameStateInvalid;
        var cur_decibel = _decibel[t];
        var cur_snr = cur_decibel - _noise_average_decibel;
        if (cur_decibel < _vad_opts.decibel_thres) {
            frame_state = FrameState.kFrameStateSil;
            DetectOneFrame(frame_state, t, false);
            return frame_state;
        }


        var sum_score = 0.0D;
        var noise_prob = 0.0D;
        Trace.Assert(_sil_pdf_ids.Length == _vad_opts.silence_pdf_num, "");
        if (_sil_pdf_ids.Length > 0) {
            Trace.Assert(_scores.GetLength(0) == 1, "只支持batch_size = 1的测试"); // 只支持batch_size = 1的测试
            var sil_pdf_scores = new float[_sil_pdf_ids.Length];
            var j = 0;
            foreach (var sil_pdf_id in _sil_pdf_ids) {
                sil_pdf_scores[j] = _scores[0, t - _idx_pre_chunk, sil_pdf_id];
                j++;
            }

            sum_score = sil_pdf_scores.Length == 0 ? 0 : sil_pdf_scores.Sum();
            noise_prob = Math.Log(sum_score) * _vad_opts.speech_2_noise_ratio;
            var total_score = 1.0D;
            sum_score = total_score - sum_score;
        }

        var speech_prob = Math.Log(sum_score);
        if (_vad_opts.output_frame_probs) {
            var frame_prob = new E2EVadFrameProbEntity();
            frame_prob.noise_prob = noise_prob;
            frame_prob.speech_prob = speech_prob;
            frame_prob.score = sum_score;
            frame_prob.frame_id = t;
            _frame_probs.Add(frame_prob);
        }

        if (Math.Exp(speech_prob) >= Math.Exp(noise_prob) + _speech_noise_thres) {
            if (cur_snr >= _vad_opts.snr_thres && cur_decibel >= _vad_opts.decibel_thres)
                frame_state = FrameState.kFrameStateSpeech;
            else
                frame_state = FrameState.kFrameStateSil;
        } else {
            frame_state = FrameState.kFrameStateSil;
            if (_noise_average_decibel < -99.9)
                _noise_average_decibel = cur_decibel;
            else
                _noise_average_decibel =
                    (cur_decibel + _noise_average_decibel * (_vad_opts.noise_frame_num_used_for_snr - 1)) /
                    _vad_opts.noise_frame_num_used_for_snr;
        }

        return frame_state;
    }

    public TimeWindow[] DefaultCall(float[,,] score, float[] waveform,
        bool isFinal = false, int maxEndSil = 800, bool online = false) {
        _max_end_sil_frame_cnt_thresh = maxEndSil - _vad_opts.speech_to_sil_time_thres;
        // compute decibel for each frame
        ComputeDecibel(waveform);
        ComputeScores(score);
        if (!isFinal)
            DetectCommonFrames();
        else
            DetectLastFrames();

        var timeWindowBatch = new List<TimeWindow>();
        if (_output_data_buf.Count > 0)
            for (var i = _output_data_buf_offset; i < _output_data_buf.Count; i++) {
                int startMs;
                int endMs;
                if (online) {
                    if (!_output_data_buf[i].contain_seg_start_point) continue;

                    if (!_next_seg && !_output_data_buf[i].contain_seg_end_point) continue;

                    startMs = _next_seg ? _output_data_buf[i].start_ms : -1;
                    if (_output_data_buf[i].contain_seg_end_point) {
                        endMs = _output_data_buf[i].end_ms;
                        _next_seg = true;
                        _output_data_buf_offset += 1;
                    } else {
                        endMs = -1;
                        _next_seg = false;
                    }
                } else {
                    if (!isFinal && (!_output_data_buf[i].contain_seg_start_point ||
                                     !_output_data_buf[i].contain_seg_end_point))
                        continue;

                    startMs = _output_data_buf[i].start_ms;
                    endMs = _output_data_buf[i].end_ms;
                    _output_data_buf_offset += 1;
                }

                timeWindowBatch.Add(new TimeWindow(
                    TimeSpan.FromMilliseconds(startMs),
                    TimeSpan.FromMilliseconds(endMs)));
            }

        if (isFinal)
            // reset class variables and clear the dict for the next query
            AllResetDetection();

        return timeWindowBatch.ToArray();
    }

    private int DetectCommonFrames() {
        if (_vad_state_machine == (int)VadStateMachine.kVadInStateEndPointDetected) return 0;

        for (var i = _vad_opts.nn_eval_block_size - 1; i > -1; i += -1) {
            var frame_state = FrameState.kFrameStateInvalid;
            frame_state = GetFrameState(_frm_cnt - 1 - i);
            DetectOneFrame(frame_state, _frm_cnt - 1 - i, false);
        }

        _idx_pre_chunk += _scores.GetLength(1) * _scores.GetLength(0); //_scores.shape[1];
        return 0;
    }

    private int DetectLastFrames() {
        if (_vad_state_machine == (int)VadStateMachine.kVadInStateEndPointDetected) return 0;

        for (var i = _vad_opts.nn_eval_block_size - 1; i > -1; i += -1) {
            var frame_state = FrameState.kFrameStateInvalid;
            frame_state = GetFrameState(_frm_cnt - 1 - i);
            if (i != 0)
                DetectOneFrame(frame_state, _frm_cnt - 1 - i, false);
            else
                DetectOneFrame(frame_state, _frm_cnt - 1, true);
        }

        return 0;
    }

    private void DetectOneFrame(FrameState cur_frm_state, int cur_frm_idx, bool is_final_frame) {
        var tmp_cur_frm_state = FrameState.kFrameStateInvalid;
        if (cur_frm_state == FrameState.kFrameStateSpeech) {
            if (Math.Abs(1.0) > _vad_opts.fe_prior_thres) //Fabs
                tmp_cur_frm_state = FrameState.kFrameStateSpeech;
            else
                tmp_cur_frm_state = FrameState.kFrameStateSil;
        } else if (cur_frm_state == FrameState.kFrameStateSil) {
            tmp_cur_frm_state = FrameState.kFrameStateSil;
        }

        var state_change = _windows_detector.DetectOneFrame(tmp_cur_frm_state, cur_frm_idx);
        var frm_shift_in_ms = _vad_opts.frame_in_ms;
        if (AudioChangeState.kChangeStateSil2Speech == state_change) {
            var silence_frame_count = _continous_silence_frame_count; // no used
            _continous_silence_frame_count = 0;
            _pre_end_silence_detected = false;
            var start_frame = 0;
            if (_vad_state_machine == (int)VadStateMachine.kVadInStateStartPointNotDetected) {
                start_frame = Math.Max(_data_buf_start_frame, cur_frm_idx - LatencyFrmNumAtStartPoint());
                OnVoiceStart(start_frame);
                _vad_state_machine = (int)VadStateMachine.kVadInStateInSpeechSegment;
                for (var t = start_frame + 1; t < cur_frm_idx + 1; t++) OnVoiceDetected(t);
            } else if (_vad_state_machine == (int)VadStateMachine.kVadInStateInSpeechSegment) {
                for (var t = _latest_confirmed_speech_frame + 1; t < cur_frm_idx; t++) OnVoiceDetected(t);

                if (cur_frm_idx - _confirmed_start_frame + 1 >
                    _vad_opts.max_single_segment_time / frm_shift_in_ms) {
                    OnVoiceEnd(cur_frm_idx, false, false);
                    _vad_state_machine = (int)VadStateMachine.kVadInStateEndPointDetected;
                } else if (!is_final_frame) {
                    OnVoiceDetected(cur_frm_idx);
                } else {
                    MaybeOnVoiceEndIfLastFrame(is_final_frame, cur_frm_idx);
                }
            } else {
                return;
            }
        } else if (AudioChangeState.kChangeStateSpeech2Sil == state_change) {
            _continous_silence_frame_count = 0;
            if (_vad_state_machine == (int)VadStateMachine.kVadInStateStartPointNotDetected) return;

            if (_vad_state_machine == (int)VadStateMachine.kVadInStateInSpeechSegment) {
                if (cur_frm_idx - _confirmed_start_frame + 1 >
                    _vad_opts.max_single_segment_time / frm_shift_in_ms) {
                    OnVoiceEnd(cur_frm_idx, false, false);
                    _vad_state_machine = (int)VadStateMachine.kVadInStateEndPointDetected;
                } else if (!is_final_frame) {
                    OnVoiceDetected(cur_frm_idx);
                } else {
                    MaybeOnVoiceEndIfLastFrame(is_final_frame, cur_frm_idx);
                }
            } else {
                return;
            }
        } else if (AudioChangeState.kChangeStateSpeech2Speech == state_change) {
            _continous_silence_frame_count = 0;
            if (_vad_state_machine == (int)VadStateMachine.kVadInStateInSpeechSegment) {
                if (cur_frm_idx - _confirmed_start_frame + 1 >
                    _vad_opts.max_single_segment_time / frm_shift_in_ms) {
                    _max_time_out = true;
                    OnVoiceEnd(cur_frm_idx, false, false);
                    _vad_state_machine = (int)VadStateMachine.kVadInStateEndPointDetected;
                } else if (!is_final_frame) {
                    OnVoiceDetected(cur_frm_idx);
                } else {
                    MaybeOnVoiceEndIfLastFrame(is_final_frame, cur_frm_idx);
                }
            } else {
                return;
            }
        } else if (AudioChangeState.kChangeStateSil2Sil == state_change) {
            _continous_silence_frame_count += 1;
            if (_vad_state_machine == (int)VadStateMachine.kVadInStateStartPointNotDetected) {
                // silence timeout, return zero length decision
                if ((_vad_opts.detect_mode == (int)VadDetectMode.kVadSingleUtteranceDetectMode &&
                     _continous_silence_frame_count * frm_shift_in_ms > _vad_opts.max_start_silence_time) ||
                    (is_final_frame && _number_end_time_detected == 0)) {
                    for (var t = _lastest_confirmed_silence_frame + 1; t < cur_frm_idx; t++) OnSilenceDetected(t);

                    OnVoiceStart(0, true);
                    OnVoiceEnd(0, true, false);
                    _vad_state_machine = (int)VadStateMachine.kVadInStateEndPointDetected;
                } else {
                    if (cur_frm_idx >= LatencyFrmNumAtStartPoint())
                        OnSilenceDetected(cur_frm_idx - LatencyFrmNumAtStartPoint());
                }
            } else if (_vad_state_machine == (int)VadStateMachine.kVadInStateInSpeechSegment) {
                if (_continous_silence_frame_count * frm_shift_in_ms >= _max_end_sil_frame_cnt_thresh) {
                    var lookback_frame = _max_end_sil_frame_cnt_thresh / frm_shift_in_ms;
                    if (_vad_opts.do_extend != 0) {
                        lookback_frame -= _vad_opts.lookahead_time_end_point / frm_shift_in_ms;
                        lookback_frame -= 1;
                        lookback_frame = Math.Max(0, lookback_frame);
                    }

                    OnVoiceEnd(cur_frm_idx - lookback_frame, false, false);
                    _vad_state_machine = (int)VadStateMachine.kVadInStateEndPointDetected;
                } else if (cur_frm_idx - _confirmed_start_frame + 1 >
                           _vad_opts.max_single_segment_time / frm_shift_in_ms) {
                    OnVoiceEnd(cur_frm_idx, false, false);
                    _vad_state_machine = (int)VadStateMachine.kVadInStateEndPointDetected;
                } else if (_vad_opts.do_extend != 0 && !is_final_frame) {
                    if (_continous_silence_frame_count <=
                        _vad_opts.lookahead_time_end_point / frm_shift_in_ms)
                        OnVoiceDetected(cur_frm_idx);
                } else {
                    MaybeOnVoiceEndIfLastFrame(is_final_frame, cur_frm_idx);
                }
            } else {
                return;
            }
        }

        if (_vad_state_machine == (int)VadStateMachine.kVadInStateEndPointDetected &&
            _vad_opts.detect_mode == (int)VadDetectMode.kVadMutipleUtteranceDetectMode)
            ResetDetection();
    }
}