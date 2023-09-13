#include <common/singal_link.h>
#include <assert.h>

/****************************************************************************
 * Name: dq_addafter
 *
 * Description:
 *  dq_addafter function adds 'node' after 'prev' in the 'queue.'
 *
 ****************************************************************************/

void dq_addafter(dq_entry_t *prev, dq_entry_t *node,
                 dq_queue_t *queue)
{
  if (!queue->head || prev == queue->tail)
    {
      dq_addlast(node, queue);
    }
  else
    {
      dq_entry_t *next = prev->flink;
      node->blink = prev;
      node->flink = next;
      next->blink = node;
      prev->flink = node;
    }
}

/****************************************************************************
 * Name: dq_addbefore
 *
 * Description:
 *   dq_addbefore adds 'node' before 'next' in 'queue'
 *
 ****************************************************************************/

void dq_addbefore(dq_entry_t *next, dq_entry_t *node,
                  dq_queue_t *queue)
{
  if (!queue->head || next == queue->head)
    {
      dq_addfirst(node, queue);
    }
  else
    {
      dq_entry_t *prev = next->blink;
      node->flink = next;
      node->blink = prev;
      prev->flink = node;
      next->blink = node;
    }
}

/****************************************************************************
 * Name: dq_addfirst
 *
 * Description:
 *  dq_addfirst affs 'node' at the beginning of 'queue'
 *
 ****************************************************************************/

void dq_addfirst(dq_entry_t *node, dq_queue_t *queue)
{
  node->blink = nullptr;
  node->flink = queue->head;

  if (!queue->head)
    {
      queue->head = node;
      queue->tail = node;
    }
  else
    {
      queue->head->blink = node;
      queue->head = node;
    }
}

/****************************************************************************
 * Name: dq_addlast
 *
 * Description
 *   dq_addlast adds 'node' to the end of 'queue'
 *
 ****************************************************************************/

void dq_addlast(dq_entry_t *node, dq_queue_t *queue)
{
  node->flink = nullptr;
  node->blink = queue->tail;

  if (!queue->head)
    {
      queue->head = node;
      queue->tail = node;
    }
  else
    {
      queue->tail->flink = node;
      queue->tail        = node;
    }
}

/****************************************************************************
 * Name: dq_cat
 *
 * Description:
 *   Move the content of queue1 to the end of queue2.
 *
 ****************************************************************************/

void dq_cat(dq_queue_t *queue1, dq_queue_t *queue2)
{
  assert(queue1 != nullptr && queue2 != nullptr);

  /* If queue2 is empty, then just move queue1 to queue2 */

  if (dq_empty(queue2))
    {
      dq_move(queue1, queue2);
    }

  /* Do nothing if queue1 is empty */

  else if (!dq_empty(queue1))
    {
      /* Attach the head of queue1 to the final entry of queue2 */

      queue2->tail->flink = queue1->head;
      queue1->head->blink = queue2->tail;

      /* The tail of queue1 is the new tail of queue2 */

      queue2->tail = queue1->tail;
      dq_init(queue1);
    }
}

/****************************************************************************
 * Name: dq_count
 *
 * Description:
 *   Return the number of nodes in the queue.
 *
 ****************************************************************************/

uint16_t dq_count(dq_queue_t *queue)
{
  dq_entry_t *node;
  uint16_t count;

  assert(queue != nullptr);

  for (node = queue->head, count = 0;
       node != nullptr;
       node = node->flink, count++);

  return count;
}

/****************************************************************************
 * Name: dq_rem
 *
 * Descripton:
 *   dq_rem removes 'node' from 'queue'
 *
 ****************************************************************************/

void dq_rem(dq_entry_t *node, dq_queue_t *queue)
{
  dq_entry_t *prev = node->blink;
  dq_entry_t *next = node->flink;

  if (!prev)
    {
      queue->head = next;
    }
  else
    {
      prev->flink = next;
    }

  if (!next)
    {
      queue->tail = prev;
    }
  else
    {
      next->blink = prev;
    }

  node->flink = nullptr;
  node->blink = nullptr;
}

/****************************************************************************
 * Name: dq_remfirst
 *
 * Description:
 *   dq_remfirst removes 'node' from the head of 'queue'
 *
 ****************************************************************************/

dq_entry_t *dq_remfirst(dq_queue_t *queue)
{
  dq_entry_t *ret = queue->head;

  if (ret)
    {
      dq_entry_t *next = ret->flink;
      if (!next)
        {
          queue->head = nullptr;
          queue->tail = nullptr;
        }
      else
        {
          queue->head = next;
          next->blink = nullptr;
        }

      ret->flink = nullptr;
      ret->blink = nullptr;
    }

  return ret;
}

/***************************************************(************************
 * Name: dq_remlast
 *
 * Description:
 *   dq_remlast removes the last entry from 'queue'
 *
 ****************************************************************************/

dq_entry_t *dq_remlast(dq_queue_t *queue)
{
  dq_entry_t *ret = queue->tail;

  if (ret)
    {
      dq_entry_t *prev = ret->blink;
      if (!prev)
        {
          queue->head = nullptr;
          queue->tail = nullptr;
        }
      else
        {
          queue->tail = prev;
          prev->flink = nullptr;
        }

      ret->flink = nullptr;
      ret->blink = nullptr;
    }

  return ret;
}

/****************************************************************************
 * Name: sq_addafter.c
 *
 * Description:
 *  The sq_addafter function adds 'node' after 'prev' in the 'queue.'
 *
 ****************************************************************************/

void sq_addafter(sq_entry_t *prev, sq_entry_t *node,
                 sq_queue_t *queue)
{
  if (!queue->head || prev == queue->tail)
    {
      sq_addlast(node, queue);
    }
  else
    {
      node->flink = prev->flink;
      prev->flink = node;
    }
}

/****************************************************************************
 * Name: sq_addfirst
 *
 * Description:
 *   The sq_addfirst function places the 'node' at the head of the 'queue'
 *
 ****************************************************************************/

void sq_addfirst(sq_entry_t *node, sq_queue_t *queue)
{
  node->flink = queue->head;
  if (!queue->head)
    {
      queue->tail = node;
    }

  queue->head = node;
}

/****************************************************************************
 * Name: sq_addlast
 *
 * Description:
 *   The sq_addlast function places the 'node' at the tail of
 *   the 'queue'
 ****************************************************************************/

void sq_addlast(sq_entry_t *node, sq_queue_t *queue)
{
  node->flink = nullptr;
  if (!queue->head)
    {
      queue->head = node;
      queue->tail = node;
    }
  else
    {
      queue->tail->flink = node;
      queue->tail        = node;
    }
}

/****************************************************************************
 * Name: sq_cat
 *
 * Description:
 *   Move the content of queue1 to the end of queue2.
 *
 ****************************************************************************/

void sq_cat(sq_queue_t *queue1, sq_queue_t *queue2)
{
  assert(queue1 != nullptr && queue2 != nullptr);

  /* If queue2 is empty, then just move queue1 to queue2 */

  if (sq_empty(queue2))
    {
      sq_move(queue1, queue2);
    }

  /* Do nothing if queue1 is empty */

  else if (!sq_empty(queue1))
    {
      /* Attach the head of queue1 to the final entry of queue2 */

      queue2->tail->flink = queue1->head;

      /* The tail of queue1 is the new tail of queue2 */

      queue2->tail = queue1->tail;
      sq_init(queue1);
    }
}

/****************************************************************************
 * Name: sq_count
 *
 * Description:
 *   Return the number of nodes in the queue.
 *
 ****************************************************************************/

uint16_t sq_count(sq_queue_t *queue)
{
  sq_entry_t *node;
  uint16_t count;

  assert(queue != nullptr);

  for (node = queue->head, count = 0;
       node != nullptr;
       node = node->flink, count++);

  return count;
}

/****************************************************************************
 * Name: sq_rem
 *
 * Description:
 *   sq_rem removes a 'node' for 'queue.'
 *
 ****************************************************************************/

void sq_rem(sq_entry_t *node, sq_queue_t *queue)
{
  if (queue->head && node)
    {
      if (node == queue->head)
        {
          queue->head = node->flink;
          if (node == queue->tail)
            {
              queue->tail = nullptr;
            }
        }
      else
        {
          sq_entry_t *prev;

          for (prev = (sq_entry_t *)queue->head;
               prev && prev->flink != node;
               prev = prev->flink);

          if (prev)
            {
              sq_remafter(prev, queue);
            }
        }
    }
}

/****************************************************************************
 * Name: sq_remafter
 *
 * Description:
 *   sq_remafter removes the entry following 'node' from the'queue'  Returns
 *   a reference to the removed entry.
 *
 ****************************************************************************/

sq_entry_t *sq_remafter(sq_entry_t *node, sq_queue_t *queue)
{
  sq_entry_t *ret = node->flink;

  if (queue->head && ret)
    {
      if (queue->tail == ret)
        {
          queue->tail = node;
          node->flink = nullptr;
        }
      else
        {
          node->flink = ret->flink;
        }

      ret->flink = nullptr;
    }

  return ret;
}

/****************************************************************************
 * Name: sq_remfirst
 *
 * Description:
 *   sq_remfirst function removes the first entry from 'queue'
 *
 ****************************************************************************/

sq_entry_t *sq_remfirst(sq_queue_t *queue)
{
  sq_entry_t *ret = queue->head;

  if (ret)
    {
      queue->head = ret->flink;
      if (!queue->head)
        {
          queue->tail = nullptr;
        }

      ret->flink = nullptr;
    }

  return ret;
}

/****************************************************************************
 * Name: sq_remlast
 *
 * Description:
 *   Removes the last entry in a singly-linked queue.
 *
 ****************************************************************************/

sq_entry_t *sq_remlast(sq_queue_t *queue)
{
  sq_entry_t *ret = queue->tail;

  if (ret)
    {
      if (queue->head == queue->tail)
        {
          queue->head = nullptr;
          queue->tail = nullptr;
        }
      else
        {
          sq_entry_t *prev;
          for (prev = queue->head;
              prev && prev->flink != ret;
              prev = prev->flink);

          if (prev)
            {
              prev->flink = nullptr;
              queue->tail = prev;
            }
        }

      ret->flink = nullptr;
    }

  return ret;
}
