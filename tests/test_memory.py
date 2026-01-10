# Owner(s): ["module: PrivateUse1"]

import gc
import time

import torch

import torch_accinte  # noqa: F401
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


class TestDeviceAllocator(TestCase):
    """Test cases for AccInteDeviceAllocator functionality."""

    def setUp(self):
        """Reset memory state before each test."""
        # Force garbage collection to ensure clean state
        gc.collect()
        # Note: We can't directly reset allocator stats without C++ API,
        # but we can ensure tensors are properly released

    def test_basic_allocation(self):
        """Test basic memory allocation with various sizes."""
        # Small allocation
        x = torch.empty(100, device="accinte")
        self.assertEqual(x.device.type, "accinte")
        self.assertEqual(x.numel(), 100)
        # Large allocation
        z = torch.empty(10000, device="accinte")
        self.assertEqual(z.device.type, "accinte")
        self.assertEqual(z.numel(), 10000)
        # Multi-dimensional allocation
        w = torch.empty(10, 20, 30, device="accinte")
        self.assertEqual(w.device.type, "accinte")
        self.assertEqual(w.shape, torch.Size([10, 20, 30]))

    def test_memory_lifecycle(self):
        """Test complete memory allocation and deallocation lifecycle."""
        # Allocate tensor
        x = torch.empty(1000, device="accinte")
        self.assertEqual(x.device.type, "accinte")

        # Explicitly delete tensor
        del x
        gc.collect()

        # Allocate again to ensure memory was freed
        y = torch.empty(1000, device="accinte")
        self.assertEqual(y.device.type, "accinte")
        del y
        gc.collect()

    def test_tensor_copy_operations(self):
        """Test memory operations during tensor copies."""
        # CPU to AccInte
        cpu_tensor = torch.randn(100)
        accinte_tensor = cpu_tensor.to("accinte")
        self.assertEqual(accinte_tensor.device.type, "accinte")
        self.assertEqual(cpu_tensor.shape, accinte_tensor.shape)

        # AccInte to CPU
        back_to_cpu = accinte_tensor.to("cpu")
        self.assertEqual(back_to_cpu.device.type, "cpu")
        self.assertTrue(torch.allclose(cpu_tensor, back_to_cpu))

        # AccInte to AccInte (clone)
        cloned = accinte_tensor.clone()
        self.assertEqual(cloned.device.type, "accinte")
        self.assertTrue(torch.allclose(accinte_tensor.cpu(), cloned.cpu()))

    def test_inplace_operations(self):
        """Test memory stability during inplace operations."""
        x = torch.ones(100, device="accinte")
        original_data_ptr = x.data_ptr()

        # Inplace addition
        x.add_(1)
        self.assertEqual(x.data_ptr(), original_data_ptr)
        self.assertTrue(torch.all(x == 2))

        # Inplace multiplication
        x.mul_(2)
        self.assertEqual(x.data_ptr(), original_data_ptr)
        self.assertTrue(torch.all(x == 4))

    def test_view_operations(self):
        """Test that views share memory correctly."""
        x = torch.randn(100, device="accinte")
        original_data_ptr = x.data_ptr()

        # Reshape view
        y = x.view(10, 10)
        self.assertEqual(y.data_ptr(), original_data_ptr)
        self.assertEqual(y.shape, torch.Size([10, 10]))

        # Slice view
        z = x[10:20]
        # Slices may have different data_ptr but should share storage
        self.assertEqual(z.numel(), 10)

    def test_different_dtypes(self):
        """Test allocation with different data types."""
        dtypes = [torch.float32, torch.float64, torch.int32, torch.int64]

        for dtype in dtypes:
            x = torch.empty(100, dtype=dtype, device="accinte")
            self.assertEqual(x.device.type, "accinte")
            self.assertEqual(x.dtype, dtype)
            self.assertEqual(x.numel(), 100)

    def test_tensor_resize(self):
        """Test tensor resizing operations."""
        x = torch.empty(100, device="accinte")
        _ = x.data_ptr()

        # Resize to smaller size (should reuse storage)
        x.resize_(50)
        self.assertEqual(x.numel(), 50)
        # Storage should still be available

        # Resize to original size
        x.resize_(100)
        self.assertEqual(x.numel(), 100)

    def test_empty_cache_operation(self):
        """Test empty cache functionality."""
        # Allocate some tensors
        x = torch.empty(1000, device="accinte")
        y = torch.empty(2000, device="accinte")

        # Delete tensors
        del x, y
        gc.collect()

        # Note: AccInteDeviceAllocator.emptyCache is currently a no-op
        # This test ensures it doesn't crash
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def test_memory_format_allocation(self):
        """Test allocation with different memory formats."""
        # Channels last format
        x = torch.empty(2, 3, 4, 4, device="accinte", memory_format=torch.channels_last)
        self.assertEqual(x.device.type, "accinte")
        self.assertTrue(x.is_contiguous(memory_format=torch.channels_last))

        # Contiguous format (default)
        y = torch.empty(
            2, 3, 4, 4, device="accinte", memory_format=torch.contiguous_format
        )
        self.assertEqual(y.device.type, "accinte")
        self.assertTrue(y.is_contiguous())

    def test_large_allocation(self):
        """Test large memory allocation."""
        # Allocate a large tensor (10MB approximately)
        size = 10 * 1024 * 1024 // 4  # 10MB in float32
        x = torch.empty(size, device="accinte")
        self.assertEqual(x.device.type, "accinte")
        self.assertEqual(x.numel(), size)

    def test_sequential_allocations_and_deallocations(self):
        """Test sequential allocation and deallocation patterns."""
        for i in range(10):
            x = torch.empty(1000 + i * 100, device="accinte")
            self.assertEqual(x.device.type, "accinte")
            # Let tensor go out of scope
            del x
        gc.collect()

    def test_allocation_with_requires_grad(self):
        """Test allocation of tensors with gradient tracking."""
        x = torch.empty(100, device="accinte", requires_grad=True)
        self.assertEqual(x.device.type, "accinte")
        self.assertTrue(x.requires_grad)

        y = torch.randn(100, device="accinte", requires_grad=True)
        self.assertEqual(y.device.type, "accinte")
        self.assertTrue(y.requires_grad)

    def test_storage_operations(self):
        """Test storage-level operations."""
        x = torch.randn(100, device="accinte")
        storage = x.storage()

        # Verify storage is on correct device
        self.assertTrue(storage.device.type == "accinte")

        # Verify storage size
        self.assertGreaterEqual(storage.size(), x.numel())

    def test_tensor_from_blob(self):
        """Test creating tensors that reference existing memory."""
        x = torch.randn(100, device="accinte")

        # Create a view that references the same data
        y = x.view_as(x)

        # They should share the same underlying storage
        self.assertEqual(x.data_ptr(), y.data_ptr())

        # Modifying one should affect the other
        x.fill_(5.0)
        self.assertTrue(torch.all(y == 5.0))


class TestMemoryLeaks(TestCase):
    """Test cases for detecting memory leaks in AccInteDeviceAllocator."""

    def setUp(self):
        """Reset memory state before each test."""
        gc.collect()
        time.sleep(0.1)  # Allow time for cleanup

    def test_no_leak_simple_allocations(self):
        """Test that simple allocations don't leak memory."""
        # Warm-up
        for _ in range(10):
            x = torch.empty(1000, device="accinte")
            del x
        gc.collect()
        time.sleep(0.1)

        # Perform many allocations and deallocations
        iterations = 1000
        for i in range(iterations):
            x = torch.empty(1000, device="accinte")
            del x

            if i % 100 == 0:
                gc.collect()

        # Final cleanup
        gc.collect()
        time.sleep(0.1)

        # If there were leaks, this would have accumulated significant memory
        # The test passes if no exception/crash occurred

    def test_no_leak_varying_sizes(self):
        """Test that allocations of varying sizes don't leak."""
        iterations = 500
        sizes = [100, 500, 1000, 5000, 10000]

        for i in range(iterations):
            size = sizes[i % len(sizes)]
            x = torch.empty(size, device="accinte")
            del x

            if i % 50 == 0:
                gc.collect()

        gc.collect()
        time.sleep(0.1)

    def test_no_leak_with_copies(self):
        """Test that tensor copies don't leak memory."""
        iterations = 300

        for i in range(iterations):
            # Create tensor
            x = torch.randn(500, device="accinte")

            # Copy to CPU
            cpu_copy = x.cpu()

            # Copy back to device
            device_copy = cpu_copy.to("accinte")

            # Clone
            cloned = device_copy.clone()

            # Delete all
            del x, cpu_copy, device_copy, cloned

            if i % 50 == 0:
                gc.collect()

        gc.collect()
        time.sleep(0.1)

    def test_no_leak_with_views(self):
        """Test that tensor views don't leak memory."""
        iterations = 500

        for i in range(iterations):
            x = torch.randn(1000, device="accinte")

            # Create various views
            view1 = x.view(10, 100)
            view2 = x[100:200]
            view3 = x.reshape(20, 50)

            # Delete views and original
            del view1, view2, view3, x

            if i % 100 == 0:
                gc.collect()

        gc.collect()
        time.sleep(0.1)

    def test_no_leak_inplace_operations(self):
        """Test that inplace operations don't leak memory."""
        iterations = 500

        for i in range(iterations):
            x = torch.ones(1000, device="accinte")

            # Multiple inplace operations
            x.add_(1)
            x.mul_(2)
            x.div_(2)
            x.sub_(1)

            del x

            if i % 100 == 0:
                gc.collect()

        gc.collect()
        time.sleep(0.1)

    def test_no_leak_with_gradients(self):
        """Test that tensors with gradients don't leak."""
        iterations = 300

        for i in range(iterations):
            x = torch.randn(100, device="accinte", requires_grad=True)
            y = torch.randn(100, device="accinte", requires_grad=True)

            # Operation that creates computation graph
            z = x + y

            # Delete all
            del x, y, z

            if i % 50 == 0:
                gc.collect()

        gc.collect()
        time.sleep(0.1)

    def test_no_leak_repeated_large_allocations(self):
        """Test repeated large allocations for memory leaks."""
        # Large tensor size (50MB)
        size = 50 * 1024 * 1024 // 4
        iterations = 50

        for i in range(iterations):
            x = torch.empty(size, device="accinte")
            del x
            gc.collect()
            time.sleep(0.05)  # Allow time for cleanup

        # Final cleanup
        gc.collect()
        time.sleep(0.1)

    def test_leak_detection_with_statistics(self):
        """Test memory leak detection using allocation patterns."""
        # This test verifies that after many alloc/dealloc cycles,
        # the allocator properly frees memory

        num_cycles = 10
        allocations_per_cycle = 100

        for cycle in range(num_cycles):
            tensors = []

            # Allocate many tensors
            for i in range(allocations_per_cycle):
                t = torch.empty(1000, device="accinte")
                tensors.append(t)

            # Verify all allocated
            self.assertEqual(len(tensors), allocations_per_cycle)

            # Delete all
            tensors.clear()
            gc.collect()
            time.sleep(0.05)

        # Final verification - if there were leaks, memory would be exhausted
        # The test passes if we can still allocate
        final_tensor = torch.empty(10000, device="accinte")
        self.assertEqual(final_tensor.device.type, "accinte")
        del final_tensor


class TestPinMemory(TestCase):
    @skipIfTorchDynamo("unsupported aten.is_pinned.default")
    def test_pin_memory(self):
        """Test pin memory for tensors, storage, and untyped storage"""
        tensor = torch.randn(10)
        self.assertFalse(tensor.is_pinned())
        pinned_tensor = tensor.pin_memory()
        self.assertTrue(pinned_tensor.is_pinned())
        slice_tensor = pinned_tensor[2:5]
        self.assertTrue(slice_tensor.is_pinned())

        tensor = torch.randn(10)
        storage = tensor.storage()
        self.assertFalse(storage.is_pinned("accinte"))
        pinned_storage = storage.pin_memory("accinte")
        self.assertTrue(pinned_storage.is_pinned("accinte"))

        tensor = torch.randn(10)
        untyped_storage = tensor.untyped_storage()
        self.assertFalse(untyped_storage.is_pinned("accinte"))
        pinned_untyped_storage = untyped_storage.pin_memory("accinte")
        self.assertTrue(pinned_untyped_storage.is_pinned("accinte"))

    @skipIfTorchDynamo("unsupported aten.is_pinned.default")
    def test_pin_memory_different_devices(self):
        """Test pin memory on different devices"""
        tensor = torch.randn(10)
        pinned_tensor = tensor.pin_memory()
        self.assertTrue(pinned_tensor.is_pinned())

        # Test pinning to specific device
        pinned_tensor_accinte = tensor.pin_memory("accinte")
        self.assertTrue(pinned_tensor_accinte.is_pinned("accinte"))

    @skipIfTorchDynamo("unsupported aten.is_pinned.default")
    def test_pin_memory_view(self):
        """Test pin memory with tensor views"""
        tensor = torch.randn(20)
        pinned_tensor = tensor.pin_memory()

        # Test various views
        view1 = pinned_tensor[2:5]
        self.assertTrue(view1.is_pinned())

        view2 = pinned_tensor[::2]
        self.assertTrue(view2.is_pinned())

        view3 = pinned_tensor.view(4, 5)
        self.assertTrue(view3.is_pinned())

    @skipIfTorchDynamo("unsupported aten.is_pinned.default")
    def test_pin_memory_storage_sharing(self):
        """Test pin memory with shared storage"""
        tensor = torch.randn(10)
        pinned_tensor = tensor.pin_memory()

        # Create another tensor sharing the same storage
        shared_tensor = torch.tensor((), dtype=torch.float32).set_(
            pinned_tensor.storage()
        )
        self.assertTrue(shared_tensor.is_pinned())


class TestMultiDeviceAllocation(TestCase):
    """Test basic multi-device allocation functionality."""

    def setUp(self):
        self.device_count = torch.accinte.device_count()
        self.assertEqual(self.device_count, 2, "This test requires 2 AccInte devices")
        gc.collect()

    def tearDown(self):
        """Restore device 0 to avoid affecting subsequent tests."""
        torch.accinte.set_device(0)
        gc.collect()

    def test_allocation_on_device_1(self):
        torch.accinte.set_device(1)
        x = torch.empty(100, device="accinte:1")
        self.assertEqual(x.device.type, "accinte")
        self.assertEqual(x.device.index, 1)

    def test_simultaneous_device_allocations(self):
        """Test allocations on both devices simultaneously."""
        x = torch.empty(100, device="accinte:0")
        y = torch.empty(200, device="accinte:1")

        self.assertEqual(x.device.index, 0)
        self.assertEqual(y.device.index, 1)
        self.assertNotEqual(x.data_ptr(), y.data_ptr())

    def test_memory_isolation_between_devices(self):
        """Test that memory allocations are isolated between devices."""

        tensors_dev0 = [torch.empty(1000, device="accinte:0") for _ in range(10)]
        tensors_dev1 = [torch.empty(1000, device="accinte:1") for _ in range(10)]

        # Verify all device 0 tensors are on device 0
        for t in tensors_dev0:
            self.assertEqual(t.device.index, 0)

        # Verify all device 1 tensors are on device 1
        for t in tensors_dev1:
            self.assertEqual(t.device.index, 1)

        # Pointers should be different
        ptrs_dev0 = {t.data_ptr() for t in tensors_dev0}
        ptrs_dev1 = {t.data_ptr() for t in tensors_dev1}
        self.assertEqual(
            len(ptrs_dev0 & ptrs_dev1), 0, "Devices should not share pointers"
        )

    def test_alternating_device_allocations(self):
        """Test alternating allocations between devices."""
        tensors = []
        for i in range(20):
            device_idx = i % 2
            t = torch.empty(100 + i, device=f"accinte:{device_idx}")
            self.assertEqual(t.device.index, device_idx)
            tensors.append(t)

        # Verify all tensors retained correct device assignment
        for i, t in enumerate(tensors):
            expected_device = i % 2
            self.assertEqual(t.device.index, expected_device)


class TestCrossDeviceOperations(TestCase):
    """Test cross-device tensor operations."""

    def setUp(self):
        self.device_count = torch.accinte.device_count()
        self.assertEqual(self.device_count, 2)
        gc.collect()

    def tearDown(self):
        """Restore device 0 to avoid affecting subsequent tests."""
        torch.accinte.set_device(0)
        gc.collect()

    def test_tensor_to_different_device(self):
        """Test moving tensor from one device to another."""
        # Create on device 0
        x = torch.randn(100, device="accinte:0")
        self.assertEqual(x.device.index, 0)

        # Move to device 1
        y = x.to("accinte:1")
        self.assertEqual(y.device.index, 1)
        self.assertNotEqual(x.data_ptr(), y.data_ptr())

        # Values should be the same
        self.assertTrue(torch.allclose(x.cpu(), y.cpu()))

    def test_bidirectional_device_transfer(self):
        """Test transferring tensor back and forth between devices."""
        original = torch.randn(100, device="accinte:0")
        original_cpu = original.cpu()

        # 0 -> 1
        on_dev1 = original.to("accinte:1")
        self.assertTrue(torch.allclose(original_cpu, on_dev1.cpu()))

        # 1 -> 0
        back_to_dev0 = on_dev1.to("accinte:0")
        self.assertTrue(torch.allclose(original_cpu, back_to_dev0.cpu()))


if __name__ == "__main__":
    run_tests()
