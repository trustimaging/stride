
from stride import Operator


class H(Operator):

    def forward(self, x):
        z = x.alike()
        # Perform some operations with x to produce z
        return z

    def adjoint(self, grad_z, x, **kwargs):
        grad_x = x.alike()
        # Calculate the gradient wrt x
        return grad_x


class G(Operator):

    def forward(self, z):
        y = z.alike()
        # Perform some operations with z to produce y
        return y

    def adjoint(self, grad_y, z, **kwargs):
        grad_z = z.alike()
        # Calculate the gradient wrt z
        return grad_z


class F(Operator):

    def forward(self, y):
        w = y.alike()
        # Perform some operations with y to produce w
        return w

    def adjoint(self, grad_w, y, **kwargs):
        grad_y = y.alike()
        # Calculate the gradient wrt y
        return grad_y


h = H()
g = G()
f = F()

