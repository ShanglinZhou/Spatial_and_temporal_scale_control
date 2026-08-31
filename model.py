# Define the RNN model

import numpy as np
import torch


# define the RNN class respecting Dale's law
class RNN_Dale:
    def __init__(self, hp):
        # load the hyperparameters
        self.task = hp['task']
        self.train_w_flag = hp['train_w_flag']
        self.train_wout_flag = hp['train_wout_flag']
        self.train_win_flag = hp['train_win_flag']

        self.N = hp['N']
        self.num_in = hp['num_In']
        self.num_out = hp['num_Out']
        self.P_inh = hp['P_inh']
        self.P_rec = hp['P_rec']
        self.w_dist = hp['w_dist']
        self.gain = hp['gain']
        self.input_gain = hp.get('input_gain', 1.0)
        self.apply_dale = hp['apply_dale']
        requested_device = str(hp.get('device', 'cpu')).lower()
        if requested_device == 'auto':
            requested_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if requested_device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
        self.device = torch.device(requested_device)

        self.sigma = hp['sigma']

        self.b_out0 = hp['b_out0']
        self.b_rec0 = hp['b_rec0']

        self.batch_size = hp['batch_size']

        self.T = hp['T']
        self.dt = hp['dt']
        self.tau_rnn = hp['tau_rnn']
        self.activation = hp['activation']
        self.dynamics_mode = hp.get('dynamics_mode', 'current_mode')
        self.rnn_type = hp.get('rnn_type', 'full_rank')
        self.rank = hp.get('rank', None)
        self.mechanism_size_input_channel = int(hp.get('mechanism_size_input_channel', 1))
        self.mechanism_speed_input_channel = int(hp.get('mechanism_speed_input_channel', 2))
        self.enforce_input_overlap = bool(hp.get('enforce_input_overlap', False))
        self.input_overlap_target = hp.get('input_overlap_target', None)
        if self.input_overlap_target is not None:
            self.input_overlap_target = float(self.input_overlap_target)
        self.output_readout_mode = str(hp.get('output_readout_mode', 'free')).lower()
        self.fixed_readout_scale = float(hp.get('fixed_readout_scale', 1.0))
        if self.rnn_type not in ['full_rank', 'low_rank']:
            raise ValueError("Unsupported rnn_type: {}. Use 'full_rank' or 'low_rank'.".format(self.rnn_type))
        if self.rnn_type == 'low_rank':
            if self.apply_dale:
                raise ValueError("low_rank RNN only supports non-Dale networks. Set hp['apply_dale'] = False.")
            if self.rank is None or int(self.rank) <= 0:
                raise ValueError("low_rank RNN requires a positive integer hp['rank'].")
            self.rank = int(self.rank)

        # Assign each unit as excitatory or inhibitory
        inh, exc, NI, NE, = self.assign_exc_inh()
        self.inh = inh
        self.exc = exc
        self.NI = NI
        self.NE = NE

        self.normal_variable = []

        # Initialize the weight matrix
        self.w_in0, self.w0, self.w_out0, self.mask, self.m_vectors0, self.n_vectors0 = self.initialize_W()

        # define variables
        if self.rnn_type == 'low_rank':
            # Compatibility tensor; the trained recurrent matrix is reconstructed from m/n factors.
            self.w = torch.tensor(self.w0, dtype=torch.float32, device=self.device)
            self.m_vectors = torch.nn.Parameter(
                torch.tensor(self.m_vectors0, dtype=torch.float32, device=self.device),
                requires_grad=self.train_w_flag,
            )
            self.n_vectors = torch.nn.Parameter(
                torch.tensor(self.n_vectors0, dtype=torch.float32, device=self.device),
                requires_grad=self.train_w_flag,
            )
        else:
            self.w = torch.nn.Parameter(torch.tensor(self.w0, dtype=torch.float32, device=self.device), requires_grad=self.train_w_flag)
            self.m_vectors = None
            self.n_vectors = None
        self.m = torch.tensor(self.mask, dtype=torch.float32, device=self.device)
        self.w_in = torch.nn.Parameter(torch.tensor(self.w_in0, dtype=torch.float32, device=self.device), requires_grad=self.train_win_flag)
        self.w_out = torch.nn.Parameter(torch.tensor(self.w_out0, dtype=torch.float32, device=self.device), requires_grad=self.train_wout_flag)

        self.b_out = torch.nn.Parameter(torch.tensor(self.b_out0, dtype=torch.float32, device=self.device), requires_grad=True)
        self.b_rec = torch.nn.Parameter(torch.tensor(self.b_rec0, dtype=torch.float32, device=self.device), requires_grad=False)

        self.train_var = [self.b_out]
        if self.train_w_flag:
            if self.rnn_type == 'low_rank':
                self.train_var.append(self.m_vectors)
                self.train_var.append(self.n_vectors)
            else:
                self.train_var.append(self.w)

        if self.train_win_flag:
            self.train_var.append(self.w_in)

        if self.train_wout_flag:
            self.train_var.append(self.w_out)

        self.tf_var = [self.w, self.w_out, self.b_out, self.m, self.w_in]
        if self.rnn_type == 'low_rank':
            self.tf_var.extend([self.m_vectors, self.n_vectors])

    def assign_exc_inh(self):

        # Apply Dale's principle
        if self.apply_dale == True:
            exc = np.zeros((self.N, 1), dtype=int)
            exc[0:np.int32(self.N * (1 - self.P_inh)), 0] = 1
            inh = 1 - exc
            NI = len(np.where(inh == True)[0])
            NE = self.N - NI
        else:
            inh = np.random.rand(self.N, 1) < 0  # no separate inhibitory units
            exc = ~inh
            NI = len(np.where(inh == True)[0])
            NE = self.N - NI
        return inh, exc, NI, NE

    def initialize_W(self):
        m_vectors0 = None
        n_vectors0 = None

        # Weight matrix
        if self.rnn_type == 'low_rank':
            factor_scale = self.gain * np.sqrt(np.sqrt(float(self.N) / float(self.rank)))
            m_vectors0 = np.float32(np.random.randn(self.N, self.rank) * factor_scale)
            n_vectors0 = np.float32(np.random.randn(self.N, self.rank) * factor_scale)
            w0 = np.matmul(m_vectors0, n_vectors0.T) / float(self.N)
            w0 = np.float32(w0)
        elif not self.apply_dale:
            w0 = np.float32(np.random.randn(self.N, self.N)) / np.sqrt(self.N) * self.gain
            if self.P_rec < 1:
                rec_mask = np.random.rand(self.N, self.N) < self.P_rec
                w0 = w0 * rec_mask.astype(np.float32)
        else:
            w0 = np.zeros((self.N, self.N), dtype=np.float32)
            idx = np.where(np.random.rand(self.N, self.N) < self.P_rec)
            if self.w_dist.lower() == 'gamma':
                # Without STP, gamma-initialized recurrent weights need explicit scaling
                # to avoid rapid rate explosion in rate_mode.
                w0[idx[0], idx[1]] = np.random.gamma(0.1, 1, len(idx[0])) * 0.25
                if self.P_rec > 0:
                    w0 = w0 / np.sqrt(self.N * self.P_rec) * self.gain
            elif self.w_dist.lower() == 'gaus':
                w0[idx[0], idx[1]] = np.random.normal(0, 1.0, len(idx[0]))
                if self.P_rec > 0:
                    w0 = w0 / np.sqrt(self.N * self.P_rec) * self.gain  # scale by a gain to make it chaotic

        if self.apply_dale == True:
            w0 = np.abs(w0)
        w0[np.diag_indices(self.N, 2)] = 0.
        # Mask matrix
        mask = np.eye((self.N), dtype=np.float32)
        mask[np.where(self.inh == True)[0], np.where(self.inh == True)[0]] = -1

        mask_ini = np.eye((self.N), dtype=np.float32)
        mask_ini[np.where(self.inh == True)[0], np.where(self.inh == True)[0]] = ((1 - self.P_inh) / self.P_inh)
        if self.apply_dale == True:
            w0 = np.matmul(w0, mask_ini)  # set E/I balance to w

        if self.output_readout_mode in ['fixed', 'frozen']:
            w_out0 = np.float32(np.random.randn(self.num_out, self.N)) / np.sqrt(self.N) * self.fixed_readout_scale
        else:
            w_out0 = np.float32(np.random.randn(self.num_out, self.N)) / np.sqrt(self.N) * self.fixed_readout_scale # 0 for previous
        if self.apply_dale == True:
            w_in0 = np.float32(np.random.gamma(0.1, 1, [self.N, self.num_in]))
        else:
            input_gain = getattr(self, 'input_gain', 1.0)
            w_in0 = np.float32(np.random.randn(self.N, self.num_in)) / np.sqrt(self.num_in) * input_gain
        w_in0 = self._impose_input_overlap_numpy(w_in0)

        return w_in0, w0, w_out0, mask, m_vectors0, n_vectors0

    def _impose_input_overlap_numpy(self, w_in0):
        if (not self.enforce_input_overlap) or self.input_overlap_target is None:
            return w_in0
        size_ch = int(self.mechanism_size_input_channel)
        speed_ch = int(self.mechanism_speed_input_channel)
        if w_in0.shape[1] <= max(size_ch, speed_ch):
            return w_in0
        rho = float(np.clip(self.input_overlap_target, -1.0, 1.0))
        eps = 1e-8
        a = np.asarray(w_in0[:, size_ch], dtype=np.float32)
        b = np.asarray(w_in0[:, speed_ch], dtype=np.float32)
        a_norm = float(np.linalg.norm(a))
        if a_norm < eps:
            a = np.float32(np.random.randn(w_in0.shape[0]))
            a_norm = float(np.linalg.norm(a))
            w_in0[:, size_ch] = a
        a_hat = a / max(a_norm, eps)
        u = b - float(np.dot(b, a_hat)) * a_hat
        u_norm = float(np.linalg.norm(u))
        if u_norm < eps:
            u = np.float32(np.random.randn(w_in0.shape[0]))
            u = u - float(np.dot(u, a_hat)) * a_hat
            u_norm = float(np.linalg.norm(u))
        u_hat = u / max(u_norm, eps)
        b_norm = float(np.linalg.norm(b))
        if b_norm < eps:
            b_norm = a_norm
        w_in0[:, speed_ch] = np.float32(b_norm * (rho * a_hat + np.sqrt(max(0.0, 1.0 - rho * rho)) * u_hat))
        return w_in0

    def enforce_input_overlap_(self):
        if (not self.enforce_input_overlap) or self.input_overlap_target is None:
            return
        size_ch = int(self.mechanism_size_input_channel)
        speed_ch = int(self.mechanism_speed_input_channel)
        if self.w_in.shape[1] <= max(size_ch, speed_ch):
            return
        rho = float(max(-1.0, min(1.0, self.input_overlap_target)))
        eps = 1e-8
        with torch.no_grad():
            a = self.w_in[:, size_ch]
            b = self.w_in[:, speed_ch]
            a_norm = torch.norm(a)
            if float(a_norm.detach().cpu()) < eps:
                return
            a_hat = a / (a_norm + eps)
            u = b - torch.sum(b * a_hat) * a_hat
            u_norm = torch.norm(u)
            if float(u_norm.detach().cpu()) < eps:
                basis = torch.zeros_like(a)
                basis[0] = 1.0
                if basis.numel() > 1 and abs(float(torch.sum(basis * a_hat).detach().cpu())) > 0.9:
                    basis[0] = 0.0
                    basis[1] = 1.0
                u = basis - torch.sum(basis * a_hat) * a_hat
                u_norm = torch.norm(u)
            u_hat = u / (u_norm + eps)
            b_norm = torch.norm(b)
            if float(b_norm.detach().cpu()) < eps:
                b_norm = a_norm
            tangent_scale = float(np.sqrt(max(0.0, 1.0 - rho * rho)))
            self.w_in[:, speed_ch].copy_(b_norm * (rho * a_hat + tangent_scale * u_hat))

    def save(self, savepath, filename):

        file = savepath + '\\' + filename + '.npy'
        VAR = []
        for i in range(len(self.tf_var)):
            VAR.append(self.tf_var[i].detach().cpu().numpy())
        np.save(file, VAR)

    def load(self, loadpath, filename):
        file = loadpath + '\\' + filename + '.npy'
        temp = np.load(file, allow_pickle=True)
        for i in range(len(self.tf_var)):
            if isinstance(self.tf_var[i], torch.nn.Parameter):
                self.tf_var[i].data.copy_(torch.tensor(temp[i], dtype=torch.float32, device=self.device))
            else:
                self.tf_var[i] = torch.tensor(temp[i], dtype=torch.float32, device=self.device)

        return self

    def recurrent_weight(self):
        if self.rnn_type == 'low_rank':
            return torch.matmul(self.m_vectors, self.n_vectors.t()) / float(self.N)
        if self.apply_dale == True:
            return torch.matmul(torch.relu(self.w), self.m)
        return self.w

    def state_dict(self):
        state = {
            'w': self.w.detach().cpu().clone(),
            'w_in': self.w_in.detach().cpu().clone(),
            'w_out': self.w_out.detach().cpu().clone(),
            'b_out': self.b_out.detach().cpu().clone(),
            'b_rec': self.b_rec.detach().cpu().clone(),
            'm': self.m.detach().cpu().clone(),
        }
        if self.rnn_type == 'low_rank':
            state['w'] = self.recurrent_weight().detach().cpu().clone()
            state['m_vectors'] = self.m_vectors.detach().cpu().clone()
            state['n_vectors'] = self.n_vectors.detach().cpu().clone()
        return state

    def load_state_dict(self, state_dict, strict=True):
        required_keys = ['w_in', 'w_out', 'b_out', 'b_rec', 'm']
        if self.rnn_type == 'low_rank':
            required_keys.extend(['m_vectors', 'n_vectors'])
        else:
            required_keys.append('w')

        if self.rnn_type == 'low_rank' and ('m_vectors' not in state_dict or 'n_vectors' not in state_dict):
            raise ValueError(
                "Cannot load a full-rank/legacy checkpoint into a low_rank model. "
                "Expected m_vectors and n_vectors in checkpoint."
            )
        if self.rnn_type == 'full_rank' and ('m_vectors' in state_dict or 'n_vectors' in state_dict):
            raise ValueError("Cannot load a low_rank checkpoint into a full_rank model.")

        if strict:
            missing = [k for k in required_keys if k not in state_dict]
            if missing:
                checkpoint_kind = 'low_rank' if (
                    'm_vectors' in state_dict or 'n_vectors' in state_dict
                ) else 'full_rank_or_legacy'
                raise KeyError(
                    'Missing keys in state_dict: {}. Model expects rnn_type={!r}; '
                    'checkpoint looks like {}.'.format(missing, self.rnn_type, checkpoint_kind)
                )

        def maybe_copy(name, tensor):
            if name in state_dict:
                tensor.data.copy_(state_dict[name].to(dtype=tensor.dtype, device=tensor.device))

        if self.rnn_type == 'low_rank':
            maybe_copy('m_vectors', self.m_vectors)
            maybe_copy('n_vectors', self.n_vectors)
        else:
            maybe_copy('w', self.w)
        maybe_copy('w_in', self.w_in)
        maybe_copy('w_out', self.w_out)
        maybe_copy('b_out', self.b_out)
        maybe_copy('b_rec', self.b_rec)
        if 'm' in state_dict:
            self.m = state_dict['m'].to(dtype=self.m.dtype, device=self.device)

    def __call__(self, stim, label, target, target_mask):
        if isinstance(stim, np.ndarray):
            stim = torch.tensor(stim, dtype=torch.float32, device=self.device)
        elif torch.is_tensor(stim):
            stim = stim.to(device=self.device, dtype=torch.float32)
        if isinstance(target, np.ndarray):
            target = torch.tensor(target, dtype=torch.float32, device=self.device)
        elif torch.is_tensor(target):
            target = target.to(device=self.device, dtype=torch.float32)
        if isinstance(target_mask, np.ndarray):
            target_mask = torch.tensor(target_mask, dtype=torch.float32, device=self.device)
        elif torch.is_tensor(target_mask):
            target_mask = target_mask.to(device=self.device, dtype=torch.float32)

        batch_size = int(stim.shape[1])
        if target.dim() == 1:
            target = target.view(1, 1, -1)
        elif target.dim() == 2:
            target = target.unsqueeze(1)
        if target_mask.dim() == 1:
            target_mask = target_mask.view(1, 1, -1)
        elif target_mask.dim() == 2:
            target_mask = target_mask.unsqueeze(1)

        x = []  # synaptic currents
        r = []  # firing-rates

        out = []  # output
        out_net = []  # output currents

        def apply_activation(v):
            if self.activation == 'sigmoid':
                return torch.sigmoid(v)
            if self.activation == 'relu':
                return torch.relu(v + self.b_rec)
            if self.activation == 'tanh':
                return torch.tanh(v + self.b_rec)
            if self.activation == 'softplus':
                return torch.clamp(torch.nn.functional.softplus(v + self.b_rec), 0, 20)
            raise ValueError("Unsupported activation: {}. Use 'sigmoid', 'relu', 'tanh' or 'softplus'.".format(self.activation))

        x.append(torch.randn(self.N, batch_size, dtype=torch.float32, device=self.device) * self.sigma)
        r.append(apply_activation(x[0]))

        out_net.append(torch.matmul(self.w_out, r[0]))
        out.append(out_net[0] + self.b_out)

        if self.rnn_type == 'low_rank':
            ww = None
        elif self.apply_dale == True:
            ww = torch.relu(self.w)  # Parametrize the weight matrix to enforce exc/inh synaptic currents
            ww = torch.matmul(ww, self.m)
        else:
            ww = self.w
        if self.apply_dale == True:
            ww_in = torch.relu(self.w_in)
        else:
            ww_in = self.w_in

        if self.dynamics_mode not in ['current_mode', 'rate_mode']:
            raise ValueError("Unsupported dynamics_mode: {}. Use 'current_mode' or 'rate_mode'.".format(self.dynamics_mode))

        # time loop
        for t in range(1, self.T):
            if self.rnn_type == 'low_rank':
                kappa = torch.matmul(self.n_vectors.t(), r[t - 1]) / float(self.N)
                recurrent_drive = torch.matmul(self.m_vectors, kappa)
            else:
                recurrent_drive = torch.matmul(ww, r[t - 1])

            inp_drive = torch.matmul(ww_in, stim[:, :, t])
            drive_noise = torch.randn(self.N, batch_size, dtype=torch.float32, device=self.device) * \
                          (self.sigma * np.sqrt(2.0 * self.tau_rnn / self.dt))

            if self.dynamics_mode == 'current_mode':
                next_x = (1 - self.dt / self.tau_rnn) * x[t - 1] + \
                         (self.dt / self.tau_rnn) * (inp_drive + recurrent_drive + drive_noise)
                x.append(next_x)
                next_r = apply_activation(next_x)
            else:
                rhs = apply_activation(recurrent_drive + inp_drive + drive_noise)
                next_r = r[t - 1] + (self.dt / self.tau_rnn) * (-r[t - 1] + rhs)
                # Guard against numerical overflow during long unrolled training.
                next_r = torch.nan_to_num(next_r, nan=0.0, posinf=20.0, neginf=0.0)
                if self.activation == 'relu':
                    next_r = torch.clamp(next_r, 0.0, 20.0)
                x.append(x[t - 1])

            r.append(next_r)

            next_o = torch.matmul(self.w_out, r[t])

            out_net.append(next_o)
            out.append(next_o + self.b_out)

        out = torch.stack(out, dim=2)
        out_net = torch.stack(out_net, dim=2)

        r = torch.stack(r, dim=2)

        loss = torch.mean(torch.square(out - target) * target_mask)
        loss = torch.sqrt(loss)

        return r, out, out_net, loss
