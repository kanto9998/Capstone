
#1
def bilinear_interpolate(self,img, scale_factor): #affine 트랜스포메이션
    n, c, h, w = img.size()
    device = img.device

    # 스케일링된 새로운 높이와 너비 계산
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)

    # affine 변환을 위한 theta 행렬 생성
    theta = torch.tensor([[scale_factor, 0, 0], [0, scale_factor, 0]], device=device)
    theta = theta.unsqueeze(0).repeat(n, 1, 1)  # 크기에 맞게 반복

    # affine_grid 생성
    affine_grid = F.affine_grid(theta, [n, c, new_h, new_w])

    # grid_sample을 사용하여 이미지 스케일링
    scaled_img = F.grid_sample(img, affine_grid)

    return scaled_img

#2
def bilinear_interpolate(self, img, scale_factor):
    print('img, scale_factor :',img,scale_factor)
    n, c, h, w = img.size()
    new_h, new_w = int(h * scale_factor), int(w * scale_factor) #뉴 WH가져오는데 여기서 미분 애매해짐
    # 디바이스 설정
    device = img.device

    # 생성할 이미지의 각 픽셀 위치에 대한 좌표 그리드 생성, 동일 디바이스로 이동
    h_scale = torch.linspace(0, h-1, new_h, device=device)
    w_scale = torch.linspace(0, w-1, new_w, device=device)

    grid_h, grid_w = torch.meshgrid(h_scale, w_scale)

    # 정수 부분과 소수 부분 분리
    h_floor = grid_h.floor().long()
    h_ceil = h_floor + 1
    h_ceil = h_ceil.clamp(max=h-1)

    w_floor = grid_w.floor().long()
    w_ceil = w_floor + 1
    w_ceil = w_ceil.clamp(max=w-1)
    print('h_floor,h_floor, h_ceil, w_floor, w_ceil :',h_floor, h_ceil, w_floor, w_ceil)

    # 4개 근접 픽셀  가져오기.. 근데 여기서도 미분이 애매해짐
    tl = img[:, :, h_floor, w_floor]
    tr = img[:, :, h_floor, w_ceil]
    bl = img[:, :, h_ceil, w_floor]
    br = img[:, :, h_ceil, w_ceil]

    # 소수 부분 계산, 동일 디바이스에서 계산
    h_frac = grid_h - h_floor.to(device)
    w_frac = grid_w - w_floor.to(device)

    # bilinear interpolation 적용
    top = tl + (tr - tl) * w_frac
    bottom = bl + (br - bl) * w_frac
    interpolated_img = top + (bottom - top) * h_frac

    return interpolated_img


#3
    class Bilinear_Interpolate(Function):
        @staticmethod
        # super().__init__()

        # self.interval = interval

        def forward(ctx, input, scale_factor):
            ctx.save_for_backward(input, scale_factor) #. df 산출식도 포함했다가 일단 공유용으로 뻄 #0301

            N, C, H, W = input.shape
            device = input.device
            print('img, scale_factor :',input,scale_factor
            # scale_factor를 사용하여 새로운 높이와 너비 계산
            new_H = torch.round(H * scale_factor).int().item()
            new_W = torch.round(W * scale_factor).int().item()
            # 새로운 좌표 생성
            h = torch.linspace(0, H-1, new_H, device=device)
            w = torch.linspace(0, W-1, new_W, device=device)
            # meshgrid 생성 <- 이렇게 해서 일단 가로세로 맞춰줌
            grid_y, grid_x = torch.meshgrid(h, w)
            # Bilinear interpolation을 위한 준비
            x0 = grid_x.floor().long()
            x1 = torch.clamp(x0 + 1, max=W-1)
            y0 = grid_y.floor().long()
            y1 = torch.clamp(y0 + 1, max=H-1)
            # 각 꼭짓점에서의 값
            Ia = input[:, :, y0, x0]
            Ib = input[:, :, y1, x0]
            Ic = input[:, :, y0, x1]
            Id = input[:, :, y1, x1]
            # 각 꼭짓점까지의 거리
            wa = (x1.float() - grid_x) * (y1.float() - grid_y)
            wb = (x1.float() - grid_x) * (grid_y - y0.float())
            wc = (grid_x - x0.float()) * (y1.float() - grid_y)
            wd = (grid_x - x0.float()) * (grid_y - y0.float())
            # wa, wb, wc, wd의 차원 변경
            wa = wa.unsqueeze(0).unsqueeze(0)  # [new_H, new_W] -> [1, 1, new_H, new_W]
            wb = wb.unsqueeze(0).unsqueeze(0)  #
            wc = wc.unsqueeze(0).unsqueeze(0)  #
            wd = wd.unsqueeze(0).unsqueeze(0)  #

            # N, C 차원에 맞게 확장
            wa = wa.expand(N, C, new_H, new_W)
            wb = wb.expand(N, C, new_H, new_W)
            wc = wc.expand(N, C, new_H, new_W)
            wd = wd.expand(N, C, new_H, new_W)
            # 이미지 합쳐가지고 만들기
            result = wa * Ia + wb * Ib + wc * Ic + wd * Id
            # 결과 이미지의 크기 조정
            result = result.permute(2, 3, 0, 1).contiguous()
            result = result.view(new_H, new_W, N, C)
            result = result.permute(2, 3, 0, 1)

            return result

        @staticmethod
        def backward(ctx, grad_output):
            input, scale_factor = ctx.saved_tensors
            N, C, H, W = input.shape
            device = input.device

            # forward에서 계산된 new_H와 new_W를 다시 계산
            new_H = torch.round(H * scale_factor).int().item()
            new_W = torch.round(W * scale_factor).int().item()
            # grad_output에 대응하는 원본 이미지의 위치를
            h = torch.linspace(0, H-1, new_H, device=device)
            w = torch.linspace(0, W-1, new_W, device=device)
            grid_y, grid_x = torch.meshgrid(h, w)
            #백워드일떄는 그냥 f interp 사용
            grad_input = F.interpolate(grad_output, size=(H, W), mode='bilinear', align_corners=True)

            #일단 포워드 확인위함
            grad_scale_factor = None

            return grad_input, grad_scale_factor

#이렇게 클래스를 만들어서 포워드랑 백워드로 구현해가지고, 클래스안에다가 DF생성모델까지 집어넣으면 그래드가 추적이 되긴했어. 일단은 df부분은 뺴놓았는데 그냥 참고만
            # 클래스 사용할떄는
                downscaled_img_newfunc = self.Bilinear_Interpolate.apply(img,1/outputs_DF)
                upscaled_img_newfunc = self.Bilinear_Interpolate.apply(downscaled_img_newfunc,outputs_DF)
